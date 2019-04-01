#!/usr/bin/env/python3
#-*- coding: utf-8 -*-


import pandas as pd
import logging
from datetime import datetime
import psql_read as p
from config import PG_CONN_1, PG_CONN_2
from params import WINDOW
import core
import sys


formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=formatter)
logger = logging.getLogger('fuel_offline')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('dut_offline.log')
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter(formatter))
logger.addHandler(fh)


def load_day_data(calc_date):
    sql = """
        select
            t.client as vehicle_id,
            extract(epoch from t.time_stamp) as time_stamp,
            t.x_msk,
            t.y_msk,
            t.latitude,
            t.longitude
        from 
            public.track_p{:%Y_%m_%d} t
    """.format(calc_date)
    track = p.psql_read(sql, PG_CONN_1)
    track.sort_values(by=['vehicle_id', 'time_stamp'], inplace=True)
    track.drop_duplicates(['vehicle_id', 'time_stamp'], inplace=True)
    track.set_index(['vehicle_id', 'time_stamp'], inplace=True)

    sql = """
            select
                t.tracker_code as vehicle_id,
                extract(epoch from t.navigation_time) as time_stamp,
                t.sensor_id,
                t.value
            from
                public.sensor_value_p{:%Y_%m_%d} t
            where value > 0
        """.format(calc_date)
    sensor_value = p.psql_read(sql, PG_CONN_1)
    sensor_value.sort_values(by=['vehicle_id', 'time_stamp'], inplace=True)

    sql = """
        select
            id as sensor_id
        from public.sensor t
        where sensor_type_id = 2
    """
    fuel_sensors = p.psql_read(sql, PG_CONN_2)

    sensor_value.drop(sensor_value.index[~sensor_value.sensor_id.isin(fuel_sensors.sensor_id)], inplace=True)
    sensor_value.drop_duplicates(['vehicle_id', 'time_stamp'], inplace=True)
    sensor_value_ = sensor_value.groupby(by=['vehicle_id', 'time_stamp']).value.sum().to_frame()
    sensor_value_['sensor_id'] = sensor_value.groupby(by=['vehicle_id', 'time_stamp']).sensor_id.min()

    return track.join(sensor_value_, how='inner')


def calc_events(vehicle_id, vehicle_data):
    logger.debug('Расчет для ТС id={}'.format(vehicle_id))
    vehicle_data = vehicle_data.sort_index()
    dist = core.calc_dist(vehicle_data.x_msk, vehicle_data.y_msk)
    vehicle_data['dist'] = dist
    events = []
    r = core.SegmentedRegression(window=30)
    df = pd.DataFrame()
    if dist[-1] > 10*1000:
        # машина должна пройти хотя бы 10 км, чтоб мы могли набрать статистику
        try:
            # простая фильтрация больших одиночных выбросов
            core.basic_filter(vehicle_data.value.values, 1)
            core.basic_filter(vehicle_data.value.values, 2)
            core.basic_filter(vehicle_data.value.values, 1)

            df = vehicle_data[['dist', 'value', 'x_msk', 'y_msk', 'latitude', 'longitude', 'sensor_id']].copy()
            df.reset_index(inplace=True)
            gb = df.groupby(by='dist')
            df = pd.concat((gb.time_stamp.min().rename('time_stamp_min'),
                            gb.time_stamp.max().rename('time_stamp_max'), gb.value.mean(),
                            gb[['x_msk', 'y_msk', 'latitude', 'longitude', 'sensor_id']].min()), axis=1).reset_index()

            r.fit(df.dist.values, df.value.values, is_sorted=True)
            events = core.splits_to_events(vehicle_id, df, r.segments_, window=2*WINDOW)
        except Exception as e:
            logger.warning('Не удалось посчитать для {}: {}'.format(vehicle_id, e))

    return events, r, df


def load_day_data_csv(path):
    df = pd.read_csv(path, index_col=['vehicle_id', 'time_stamp'], sep=';', decimal=',')
    if 'sensor_id' not in df.columns:
        df['sensor_id'] = 0
    if 'latitude' not in df.columns:
        df['latitude'] = 0
    if 'longitude' not in df.columns:
        df['longitude'] = 0
    return df


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    if len(sys.argv) > 1:
        calc_date = sys.argv[1]
    else:
        print('Введите дату в формате YYYY-MM-DD')
        calc_date = input()

    try:
        calc_date = datetime.strptime(calc_date, '%Y-%m-%d')
    except:
        raise ValueError('Не удалось распознать дату, введите в формате YYYY-MM-DD')

    data = load_day_data(calc_date)  # type: pd.DataFrame
    #data.to_csv('data_2018-08-17.csv', sep=';', decimal=',')

    for vehicle_id, vehicle_data in data.groupby(level='vehicle_id'):
        events, r, vehicle_group_df = calc_events(vehicle_id, vehicle_data)
        for e in events:
            e.dump_db(False)
            logger.info(str(e))
        core.segment_dump_db(r, vehicle_id, vehicle_group_df)
