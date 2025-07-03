import pandas as pd
import numpy as np

def process_shipment_data_for_consolidation(shipment_df: pd.DataFrame) -> pd.DataFrame:
    """ This function processes the shipment data for consolidation.
    It renames columns, groups the data, and filters out rows with zero pallets.
    Args:
        shipment_df (pd.DataFrame): The shipment data to be processed.
    Returns:
        pd.DataFrame: The processed shipment data.
    """

    rename_dict_shipment = {"PROD TYPE":"PROD TYPE","CUSTOMER_NAME":"CUSTOMER_NAME","SHIP_TO_NAME":"NAME","ORDER_ID":"ORDER_ID","DELIVERY_DATE":"SHIPPED_DATE","SHORT_POSTCODE":"SHORT_POSTCODE",
    "PALLET_DISTRIBUTION":'Total Pallets',"POSTCODE":"POSTCODE","DISTANCE":"Distance"}
    insight_data = shipment_df.rename(columns=rename_dict_shipment)
    data_for_consolidation = insight_data[list(rename_dict_shipment.values())+['SALES']]
    df = data_for_consolidation.groupby(['PROD TYPE','CUSTOMER_NAME','NAME','ORDER_ID','SHIPPED_DATE','SHORT_POSTCODE','POSTCODE','Distance'])[['Total Pallets','SALES']].sum().reset_index()
    df['ORDER_ID'] = df['ORDER_ID'].astype(int).astype(str)
    df['SHIPPED_DATE'] = pd.to_datetime(df['SHIPPED_DATE'], format='%Y-%m-%d')
    df['Total Pallets'] = np.ceil(df['Total Pallets']).astype(int)
    df = df[df['Total Pallets'] > 0]
    
    return df