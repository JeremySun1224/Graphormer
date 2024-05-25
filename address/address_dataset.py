# -*- coding: utf-8 -*-
# -*- author: jeremysun1224 -*-

from clickhouse_driver import Client

from config import CK_CONFIG

ck_client = Client(**CK_CONFIG)

train_sample_sql = """
    SELECT waybill_no,
           receiver_province_name,
           receiver_city_name,
           receiver_area_name,
           receiver_detailed_address,
           receiver_full_address,
           network_code,
           tail_code,
           station_unique_number,
           staff_code,
           station_type,
           geo_coordinate,
           poi_coordinate,
           last_sign_time,
           dt,
           ck_dt
    FROM ai_group.train_sample_station
    WHERE pick_network_code != network_code
      AND sign_finance_code in ('110000')
      AND receiver_area_name = '海淀区'
      AND ck_dt = '20230925'
    LIMIT 100;
"""  # 北京代理区为例


if __name__ == '__main__':
    _train_data = ck_client.query_dataframe(query=train_sample_sql)
    print(_train_data.head())
