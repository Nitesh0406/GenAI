import pandas as pd

csv_path = "src/data/Alloga+Delivery_Data_with_Dis.csv"
schema_path = "src/data/data_schema.xlsx"
# kpi_path = "src/data/Perrigo - Scoping&Strategies_old.xlsx"
save_figure_path = './plots'

df = pd.read_csv(csv_path)
df = df.rename(columns={'Batch': 'BATCH',
                           'CONSIGNMENT DELIVERY POINT': 'CONSIGNMENT_DELIVERY_POINT',
                           'CS CUSTOMER':'CS_CUSTOMER',
                            'NAME':'COMPANY_NAME',
                           'CUSTOMER GROWTH SEGMENTS':'CUSTOMER_GROWTH_SEGMENTS',
                           'Cost Attribution':'TRANSPORT_COST',
                           'Delivered Qty':'DELIVERED_QTY',
                           'Delivery Date':'DELIVERY_DATE',
                           'Del qty bu':'DEL_QT_ BU',
                           'Description':'DESCRIPTION',
                           'Distance':'DISTANCE',
                           'Fixed Cost':'FIXED_COST',
                           'FL Leftover':'FL_LEFTOVER',
                           'FL Units':'FL_UNITS',
                           'Footprint':'FOOTPRINT',
                           'FP Leftover':'FP_LEFTOVER',
                           'FP Units':'FP_UNITS',
                           'FS Units':'FS_UNITS',
                           'Full Layers':'FULL_LAYERS',
                           'Full Pallets':'FULL_PALLETS',
                           'Full Shippers':'FULL_SHIPPERS',
                           'LYR/PLT':'LYR/PLT',
                           'Material':'MATERIAL',
                           'NUM_LINES':'NUM_LINES',
                           'Ordered Qty':'ORDERED_QTY',
                           'Pallet Count':'PALLET_COUNT',
                           'Pallet Distribution':'PALLET_DISTRIBUTION',
                           'Net amt':'SALES',
                           'Rate':'RATE',
                           'Reference document':'REFERENCE_DOCUMENT',
                           'Sales Document':'SALES_DOCUMENT',
                           'Sales Unit':'SALES_UNIT',
                           'Ship-to':'SHIP_TO',
                           'Ship-to Name':'SHIP_TO_NAME',
                           'SHP HEIGHT':'SHP_HEIGHT',
                           'SHP LENGTH':'SHP_LENGTH',
                           'SHP WEIDTH':'SHP_WIDTH',
                           'Sold-to':'SOLD_TO',
                           'Sold to code':'SOLD_TO_CODE',
                           'Sold-to Name':'SOLD_TO_NAME',
                           'Units Picking':'UNITS_PICKING',
                           'Warehouse Cost':'WAREHOUSE_COST',
                           'PROD TYPE':'PROD_TYPE',
                           'Created On':'CREATED_ON',
                           'PLT HEIGHT':'PLT_HEIGHT'})

useful_columns = ['POSTCODE', 'ORDER_ID', 'SHORT_POSTCODE','WAREHOUSE_COST',
                  'SOLD_TO_NAME', 'DELIVERY_DATE',
       'MATERIAL', 'DESCRIPTION', 'SHIP_TO_NAME', 'SALES',
       'ORDERED_QTY', 'DELIVERED_QTY', 'CS_CUSTOMER',
       'CUSTOMER_GROWTH_SEGMENTS', 'PALLET_DISTRIBUTION',
       'TRANSPORT_COST', 'FIXED_COST', 'FOOTPRINT','DISTANCE']

df = df[useful_columns]
SAMPLE_DATA = df.head(3)

SCHEMA_df = pd.read_excel(schema_path, sheet_name = 'data_schema')
SCHEMA = SCHEMA_df.set_index('Column Name', inplace=False).to_dict()

KPI_DESC_df = pd.read_excel(schema_path, sheet_name='kpi_desc')
KPI_DESC = KPI_DESC_df.set_index('KPI Name', inplace=False).to_dict()
