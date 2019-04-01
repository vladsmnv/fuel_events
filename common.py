#-*- coding: utf-8 -*-


from config import PG_CONN_FASTTRACK_WRITE
import psycopg2
from datetime import datetime, timezone
import numpy as np
import logging
logger = logging.getLogger('fuel_offline')


#region TimePoint
class TimePoint:
    __slots__ = ('vehicle_id', 'time_stamp', 'fuel', 'latitude', 'longitude', 'x', 'y', 'speed',
                 'hi_fwd', 'lo_fwd', 'sensors')

    def __init__(self, vehicle_id, time_stamp, fuel, latitude, longitude, x, y, speed, sensors=tuple()):
        """
        Точка соответствует данным за один момент времени
        :param vehicle_id: id транспортного средства (client_id в БД)
        :param time_stamp: время в сек
        :param fuel: уровень топлива, л
        :param latitude: широта в градусах
        :param longitude: широта в градусах
        :param x: x координата msk м
        :param y: y координата msk м
        :param speed: скорость км/час
        :param sensors: id сенсоров ДУТ (список, т.к. может быть несколько)
        """
        self.vehicle_id = vehicle_id
        self.time_stamp = float(time_stamp)
        self.fuel = fuel

        self.latitude = latitude
        self.longitude = longitude
        self.x = x
        self.y = y
        self.speed = speed
        self.sensors = sensors

        self.hi_fwd = float('nan')
        self.lo_fwd = float('nan')

    def is_valid(self):
        # Проверка что есть корректные данные по уровню топлива.
        # Некоректными считаются 0, бесконечные значения, а также отсутствие значения.
        return np.isfinite(self.fuel) and self.fuel != 0.0
# endregion


class FuelEvent:
    # Информация о событии (заправке или сливе топлива)
    def __init__(self, vehicle_id, dir):
        self.vehicle_id = vehicle_id
        self.dir = dir
        self.start_point = None
        self.end_point = None
        self.start_fuel = None
        self.end_fuel = None
        self.nearest_station_dist = None
        self.event_id = None  # присваивается снаружи
        self.finished = False
        self.start_dist = None  # дистанция на начало события для удаления сливов, которые вписываются в норматив расхода
        self.end_dist = None  # дистанция на конец события для удаления сливов, которые вписываются в норматив расхода
        self.is_cancelled = False  # дистанция на начало события для удаления сливов, которые вписываются в норматив расхода

    @property
    def fuel_change(self):
        return self.end_fuel - self.start_fuel

    @property
    def dir_str(self):
        if self.dir == 1:
            return 'refill'
        else:
            return 'leak'

    def __str__(self):
        return '{}: id={}, изменение: {:.1f}л. Время: {:%Y-%m-%d %H:%M:%S} - {:%Y-%m-%d %H:%M:%S}'.format(self.dir_str, self.vehicle_id, self.fuel_change,
                                   datetime.fromtimestamp(self.start_point.time_stamp, tz=timezone.utc),
                                   datetime.fromtimestamp(self.end_point.time_stamp, tz=timezone.utc))


    def dump_db(self, is_online):
        try:
            conn = psycopg2.connect(
                host=PG_CONN_FASTTRACK_WRITE['host'],
                port=PG_CONN_FASTTRACK_WRITE['port'],
                dbname=PG_CONN_FASTTRACK_WRITE['database'],
                user=PG_CONN_FASTTRACK_WRITE['user']
            )
            cur = conn.cursor()

            if self.is_cancelled and self.event_id is not None:
                sql = 'delete from public.events where id = %s'
                cur.execute(sql, [self.event_id])
            else:
                sql_params = {
                    'event_id': self.event_id,
                    'vehicle_id': self.vehicle_id,
                    'started_at': self.start_point.time_stamp if self.start_point else None,
                    'finished_at': self.end_point.time_stamp if self.end_point else None,
                    'event_type': self.dir_str,
                    'start_lat': self.start_point.latitude if self.start_point else None,
                    'start_lon': self.start_point.longitude if self.start_point else None,
                    'start_x_msk': self.start_point.x if self.start_point else None,
                    'start_y_msk': self.start_point.y if self.start_point else None,
                    'end_lat': self.end_point.latitude if self.end_point else None,
                    'end_lon': self.end_point.longitude if self.end_point else None,
                    'end_x_msk': self.end_point.x if self.end_point else None,
                    'end_y_msk': self.end_point.y if self.end_point else None,
                    'start_value': self.start_fuel,
                    'finish_value': self.end_fuel,
                    'open_left': not self.finished,
                    'open_right': not self.finished,
                    'sensor_id': int(min(self.start_point.sensors)) if self.start_point.sensors else 0,
                    'is_online': is_online
                }
                if self.event_id is None:
                    sql = """
                        insert into public.events
                            (
                                tracker_code,
                                started_at,
                                finished_at,
                                event_type,
                                start_lat,
                                start_lon,
                                start_x_msk,
                                start_y_msk,
                                finish_lat,
                                finish_lon,
                                finish_x_msk,
                                finish_y_msk,
                                start_value,
                                finish_value,
                                open_left,
                                open_right,
                                sensor_id,
                                is_online
                            )
                        values
                            (
                                %(vehicle_id)s, 
                                to_timestamp(%(started_at)s) at time zone 'UTC',
                                to_timestamp(%(finished_at)s) at time zone 'UTC',
                                %(event_type)s,
                                %(start_lat)s,
                                %(start_lon)s,
                                %(start_x_msk)s,
                                %(start_y_msk)s,
                                %(end_lat)s,
                                %(end_lon)s,
                                %(end_x_msk)s,
                                %(end_y_msk)s,
                                %(start_value)s,
                                %(finish_value)s,
                                %(open_left)s,
                                %(open_right)s,
                                %(sensor_id)s,
                                %(is_online)s
                            )
                        returning id
                    """
                    cur.execute(sql, sql_params)
                    self.event_id = cur.fetchone()[0]
                else:
                    sql = """
                        update public.events
                        set
                                started_at = to_timestamp(%(started_at)s) at time zone 'UTC',
                                finished_at = to_timestamp(%(finished_at)s) at time zone 'UTC',
                                start_lat = %(start_lat)s,
                                start_lon = %(start_lon)s,
                                start_x_msk = %(start_x_msk)s,
                                start_y_msk = %(start_y_msk)s,
                                finish_lat = %(end_lat)s,
                                finish_lon = %(end_lon)s,
                                finish_x_msk = %(end_x_msk)s,
                                finish_y_msk = %(end_y_msk)s,
                                start_value = %(start_value)s,
                                finish_value = %(finish_value)s,
                                open_left = %(open_left)s,
                                open_right = %(open_right)s,
                                updated_at = default
                        where id = %(event_id)s
                    """
                    cur.execute(sql, sql_params)
            conn.commit()
            cur.close()
            conn.close()
        except:
            logger.error('Не удалось записать информацию об участке пути в БД. ТС id={}'.format(self.vehicle_id))
            raise
