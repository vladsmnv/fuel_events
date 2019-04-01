#-*- coding: utf-8 -*-


import subprocess
from io import BytesIO
import logging
import pandas as pd
import os
import hashlib
from config import PSQL_DIR, TEMP_FOLDER


logger = logging.getLogger('dbread')


def psql_read(sql, conn_data, date_cols=False, index_col=None, use_file=True):

    sql = sql.encode()

    file = os.path.join(TEMP_FOLDER, hashlib.md5(sql).hexdigest())
    if not os.path.exists(TEMP_FOLDER) and TEMP_FOLDER != '':
        os.makedirs(TEMP_FOLDER)

    cmd = PSQL_DIR + ('psql -h {host} -p {port} -d {database} '
                      '-U {user} '
                      '-A -P footer=off -v FETCH_COUNT=10000 -f -'
                      ).format(**conn_data)
    spparams = {
        'input': sql,
        'stdout': subprocess.PIPE,
        'stderr': subprocess.PIPE
    }

    if use_file:  # но файла нет
        logger.debug('Loading data from Vertica to file')
        cmd += ' -o {}'.format(file)  # save to physical file
        proc = subprocess.run(cmd.split(' '), **spparams)
        if len(proc.stderr):
            if os.path.isfile(file):
                os.remove(file)
            raise ValueError(proc.stderr.decode())
    else:
        logger.debug('Loading data from Postgresql to stdin')
        proc = subprocess.run(cmd.split(' '), **spparams)
        file = BytesIO(proc.stdout)

    logger.debug('Loading from db finished')

    try:
        df = pd.read_csv(file, sep='|',
                         parse_dates=date_cols, index_col=index_col)
    except:
        raise
    finally:
        if use_file:
            os.remove(file)
        else:
            file.close()
    logger.debug('File reading finished')

    assert df.index.is_unique, "Найдены дубликаты"
    return df
