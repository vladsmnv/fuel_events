#-*- coding: utf-8 -*-


import numpy as np
import itertools
import psycopg2
from config import PG_CONN_FASTTRACK_WRITE
from params import THRESHOLD_FUEL_DIFF, THRESHOLD_SLOPE
import common as cm
from main import logger
FLOAT_PRECISION = 1e-14


def get_variance_without_linear(y, x):
    cov = np.cov(y, x)
    return cov[0, 0] - (cov[1, 0] * cov[1, 0] / cov[1, 1] if cov[1, 1] > FLOAT_PRECISION else 0.)


def basic_filter(f, n=1):
    for i, (diff_m1, v, diff_p1) in enumerate(zip(f[n:]-f[:-n], f[n:], f[n:-n]-f[n*2:]), 1):
        if abs(diff_m1) > 5 and abs(diff_p1) > 5 and (diff_m1>0)==(diff_p1>0):
            f[i] = v - (diff_m1 + diff_p1) / 2


def calc_dist(x, y):
    ds = (x.diff()**2 + y.diff()**2).pow(1./2).values
    ds[np.isnan(ds)] = 0.
    ds[ds > 500] = 0.
    ds[0] = 0.
    return np.cumsum(ds)


class Segment:
    def __init__(self, idx_start, idx_end, x_start, x_end, slope, intercept):
        self.idx_start = idx_start
        self.idx_end = idx_end
        self.x_start = x_start
        self.x_end = x_end
        self.slope = slope
        self.intercept = intercept
        self.is_event = False


class SegmentedRegression:
    """Segmented linear regression model.

    Parameters
    ----------
    window : int, optional
        limit of distance in points of two sequential break points

    eps : float, optional
        threshold when stop further split search
    """

    def __init__(self, window=15, eps=None):
        # origin data
        self.x = None
        self.y = None

        # cumulative sum data
        self._n = None
        self._x = None
        self._y = None
        self._xy = None
        self._x2 = None
        self._y2 = None

        # result
        self.segments_ = []

        # model parameters
        self.eps = eps
        self.window = int(window)

    def fit(self, x, y, is_sorted=False):
        """Find segments and apply linear fit
        :param x : 1d array, Data
        :param y : 1d array, Target
        :param is_sorted : boolean, (default=False) Whether x is already sorted
        """
        assert len(x.shape) == 1, 'input data x expects to be 1d array'
        assert len(y.shape) == 1, 'input data y expects to be 1d array'
        assert x.shape[0] == y.shape[0], 'input data must have same length'
        assert x.var() != 0.0, 'input data has to have distinct values'

        if not is_sorted:
            idx = np.argsort(x)
            x, y = x[idx], y[idx]

        self.x = x
        self.y = y

        self._n = np.arange(1, x.shape[0] + 1)
        self._x = np.cumsum(x)
        self._y = np.cumsum(y)
        self._xy = np.cumsum(x * y)
        self._x2 = np.cumsum(x * x)
        self._y2 = np.cumsum(y * y)

        if self.eps is None:
            self.eps = 3 * self._estimate_var(x, y)

        self.segments_ = self._find_segments(0, x.shape[0], self.window, self.eps)

    def _find_segments(self, n1, n2, window, eps):
        n, v, v_r = self._get_variance_slice(n1, n2)

        if (n - n1 <= window) or (n + window >= n2):
            return [self._get_segment_info(n1, n2)]
        else:
            if v < eps and v_r < eps:
                return [self._get_segment_info(n1, n), self._get_segment_info(n, n2)]
            elif v >= eps and v_r < eps:
                return self._find_segments(n1, n, window, eps) + [self._get_segment_info(n, n2), ]
            elif v < eps and v_r >= eps:
                return [self._get_segment_info(n1, n), ] + self._find_segments(n, n2, window, eps)
            else:
                return self._find_segments(n1, n, window, eps) + \
                       self._find_segments(n, n2, window, eps)

    def _get_segment_info(self, n1, n2):
        x, y = self.x, self.y
        cov = np.cov(x[n1:n2], y[n1:n2])
        if abs(cov[0, 0]) < FLOAT_PRECISION:
            k = np.inf
            b = np.nan
        else:
            k = cov[0, 1] / cov[0, 0]
            b = y[n1:n2].mean() - k * x[n1:n2].mean()
        return Segment(n1, n2, x[n1], x[n2-1], k, b)

    def _get_variance_slice(self, n1, n2):
        x, y, xy, x2, y2, n = self._x, self._y, self._xy, self._x2, self._y2, self._n

        n_ = n[n1: n2] - (n[n1] - 1)
        x_m = (x[n1: n2] - x[n1]) / n_
        y_m = (y[n1: n2] - y[n1]) / n_
        xy_m = (xy[n1: n2] - xy[n1]) / n_
        cov = xy_m - x_m * y_m

        x2_m = (x2[n1: n2] - x2[n1]) / n_
        var_x = x2_m - x_m * x_m
        var_x[var_x < FLOAT_PRECISION] = np.nan  # to avoid division by zero

        y2_m = (y2[n1: n2] - y2[n1]) / n_
        var_y = y2_m - y_m * y_m

        v = var_y - cov * cov / var_x
        v[0] = np.nan  # one point estimation is rubbish

        n2 -= 1  # last element will be added in the end
        n_ = n[n2] - n[n1: n2]
        x_m = (x[n2] - x[n1: n2]) / n_
        y_m = (y[n2] - y[n1: n2]) / n_
        xy_m = (xy[n2] - xy[n1: n2]) / n_
        cov = xy_m - x_m * y_m

        x2_m = (x2[n2] - x2[n1: n2]) / n_
        var_x = x2_m - x_m * x_m
        var_x[var_x < FLOAT_PRECISION] = np.nan

        y2_m = (y2[n2] - y2[n1: n2]) / n_
        var_y = y2_m - y_m * y_m

        v_r = np.zeros_like(v)
        v_r[1:] = var_y - cov * cov / var_x

        try:
            n_relative = np.nanargmin(v + v_r)
            return n1 + n_relative, v[n_relative], v_r[n_relative]
        except:
            # if all values is nan
            return n1, 0, 0

    @staticmethod
    def _estimate_var(x, y):
        len_x = len(x)
        n = max(min(1000, len_x // 10), 100)

        if n >= len_x:
            return get_variance_without_linear(y, x)

        slice_list =[list(zip(range((n//4 )*i, len_x, n), range(n + (n//4 )*i, len_x, n))) for i in range(4)]
        slice_list = itertools.chain.from_iterable(slice_list)
        var = [
            get_variance_without_linear(y[i1:i2], x[i1:i2])
            for i1, i2 in slice_list
        ]

        return np.percentile(var, 20)

    def predict(self, x):
        segments = self.segments_
        if len(segments) == 0:
            return 0 * x
        elif len(segments) == 1:
            s = segments[0]
            return s.slope * x + s.intercept
        else:
            condlist = [x < segments[1].x_start, ] + [
                x >= segment.x_start
                for segment in segments[1:]
            ]
            funclist = [
                lambda x, k=s.slope, b=s.intercept: k*x + b
                for s in segments
            ]
            return np.piecewise(x, condlist, funclist)


def splits_to_events(vehicle_id, vehicle_data, lines, window):

    fuel = vehicle_data.value
    dist = vehicle_data.dist.values
    events = []

    # базовые события - большие разрывы в графике потребления топлива
    # последний элемент s_next = None нужен для проверки на резкий наклон самого последнего сегмента
    for s_prev, s_next in zip(lines, lines[1:] + [None]):  # type: Segment
        # огромный наклон == событие
        if s_prev.slope * 100000 < - THRESHOLD_SLOPE or s_prev.slope * 100000 > THRESHOLD_SLOPE/5:
            s_prev.is_event = True
            idx_start, idx_end = s_prev.idx_start, s_prev.idx_end-1
            y_left = fuel.iat[idx_start]
            y_right = fuel.iat[idx_end]
            if abs(y_right - y_left) > THRESHOLD_FUEL_DIFF:
                if y_right > y_left:
                    e = cm.FuelEvent(vehicle_id, 1)
                else:
                    e = cm.FuelEvent(vehicle_id, -1)

                e.start_fuel = y_left
                e.start_point = cm.TimePoint(vehicle_id,
                                             vehicle_data.time_stamp_max.iat[idx_start],
                                             y_left,
                                             vehicle_data.latitude.iat[idx_start],
                                             vehicle_data.longitude.iat[idx_start],
                                             vehicle_data.x_msk.iat[idx_start],
                                             vehicle_data.y_msk.iat[idx_start],
                                             0.0, (vehicle_data.sensor_id.iat[idx_start],))
                e.end_fuel = y_right
                e.end_point = cm.TimePoint(vehicle_id,
                                           vehicle_data.time_stamp_min.iat[idx_end],
                                           y_right,
                                           vehicle_data.latitude.iat[idx_end],
                                           vehicle_data.longitude.iat[idx_end],
                                           vehicle_data.x_msk.iat[idx_end],
                                           vehicle_data.y_msk.iat[idx_end],
                                           0.0, (vehicle_data.sensor_id.iat[idx_start],))

                events.append(e)

        if s_next is None:
            break

        idx_prev, idx, left_slope, left_intercept = s_prev.idx_start, s_prev.idx_end, s_prev.slope, s_prev.intercept
        idx_next, right_slope, right_intercept = s_next.idx_end, s_next.slope, s_next.intercept
        # проверям разрыв ли это
        idx_w = max(idx-window, 0)
        fuel_values = fuel.iloc[idx_w:idx+window].values
        high_v = np.percentile(fuel_values, 90)
        low_v = np.percentile(fuel_values, 10)
        if abs(high_v - low_v) > THRESHOLD_FUEL_DIFF:
            curr_x = dist[idx]
            y_left = left_slope * curr_x + left_intercept
            y_right = right_slope * curr_x + right_intercept

            if abs(y_right - y_left) > THRESHOLD_FUEL_DIFF:
                idx_h = np.argmax(fuel_values[fuel_values < high_v])
                idx_l = np.argmin(fuel_values[fuel_values > low_v])
                if y_right > y_left:
                    e = cm.FuelEvent(vehicle_id, 1)
                    idx_start = idx_w + idx_l
                    idx_end = idx_w + idx_h
                else:
                    e = cm.FuelEvent(vehicle_id, -1)
                    idx_start = idx_w + idx_h
                    idx_end = idx_w + idx_l
                idx_end = max(idx_end, idx_start)  # на всякий случай

                e.start_fuel = y_left
                e.start_point = cm.TimePoint(
                    vehicle_id,
                    vehicle_data.time_stamp_max.iat[idx_start],
                    y_left,
                    vehicle_data.latitude.iat[idx_start],
                    vehicle_data.longitude.iat[idx_start],
                    vehicle_data.x_msk.iat[idx_start],
                    vehicle_data.y_msk.iat[idx_start],
                    0.0, (vehicle_data.sensor_id.iat[idx_start],)
                )
                e.end_fuel = y_right
                e.end_point = cm.TimePoint(
                    vehicle_id,
                    vehicle_data.time_stamp_min.iat[idx_end],
                    y_right,
                    vehicle_data.latitude.iat[idx_end],
                    vehicle_data.longitude.iat[idx_end],
                    vehicle_data.x_msk.iat[idx_end],
                    vehicle_data.y_msk.iat[idx_end],
                    0.0, (vehicle_data.sensor_id.iat[idx_end],)
                )
                e.finished = True
                events.append(e)


    # если события близки - объеденить их в одно
    # events.sort(key=lambda x: x.start_point.time_stamp) в сортировке не нуждается, их порядок уже верен,
    # важно, чтобы порядок не нарушился из-за разницы алгоритмов
    while True:
        # внешний цикл нужен чтоб сливать более двух событий
        for i, (e_prev, e_next) in enumerate(zip(events[:-1], events[1:])):  # type: int, FuelEvent, FuelEvent
            # два события подряд - по сути одно
            if e_prev.dir == e_next.dir and e_next.start_point.time_stamp - e_prev.end_point.time_stamp < 60*10:
                e_prev.end_point = e_next.end_point
                e_prev.end_fuel = e_next.end_fuel
                break
        else:
            # не нашли пары близких событий - заканчиваем
            break

        events.pop(i+1)

    return events

def segment_dump_db(r, vehicle_id, vehicle_data):
    try:
        conn = psycopg2.connect(
            host=PG_CONN_FASTTRACK_WRITE['host'],
            port=PG_CONN_FASTTRACK_WRITE['port'],
            dbname=PG_CONN_FASTTRACK_WRITE['database'],
            user=PG_CONN_FASTTRACK_WRITE['user']
        )
        cur = conn.cursor()
        for segment in r.segments_:
            if segment.is_event:
                if segment.slope > 0:
                    event_type = 'refill'
                else:
                    event_type = 'leak'
            else:
                event_type = None

            x_start = vehicle_data.dist.iat[segment.idx_start]
            x_end = vehicle_data.dist.iat[segment.idx_end - 1]
            sql_params = {
                'vehicle_id': vehicle_id,
                'started_at': int(vehicle_data.time_stamp_min.iat[segment.idx_start]),
                'finished_at': int(vehicle_data.time_stamp_max.iat[segment.idx_end - 1]),
                'fuel_rate': - segment.slope * 1e5,
                'distance': (x_end - x_start)/1e3,
                'start_fuel': segment.slope * x_start + segment.intercept,
                'finish_fuel': segment.slope * x_end + segment.intercept,
                'std_dev': np.sqrt(r.eps/3),
                'event_type': event_type
            }
            sql = """
                insert into public.fuel_rate
                    (
                        tracker_code,
                        started_at,
                        finished_at,
                        fuel_rate,
                        distance,
                        start_fuel,
                        finish_fuel,
                        std_dev,
                        event_type
                    )
                values
                    (
                        %(vehicle_id)s, 
                        to_timestamp(%(started_at)s) at time zone 'UTC',
                        to_timestamp(%(finished_at)s) at time zone 'UTC',
                        %(fuel_rate)s,
                        %(distance)s,
                        %(start_fuel)s,
                        %(finish_fuel)s,
                        %(std_dev)s,
                        %(event_type)s
                    )
            """
            cur.execute(sql, sql_params)
        conn.commit()
        cur.close()
        conn.close()
    except:
        logger.warning('Не удалось записать информацию об участке пути в БД. ТС id={}'.format(vehicle_id))
        raise
