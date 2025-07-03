
def generate_description_and_followup_question(question):
    if question == "Provide me some examples of Insight Questions":

        description = """
**Insight Agent** is a data analytics tool designed to help users explore trends, compare performance, and answer common business questions related to logistics and operations. It simplifies access to key information by bringing together data from multiple sources and presenting it in a clear, structured format. The tool supports better decision-making by highlighting patterns, identifying changes over time, and enabling quick benchmarking across different dimensions.

**Key KPIs Example:**
- **Transport Cost**
- **Sales** 
- **Total Orders**
- **Pallet per Order**
- **Cost per Pallet**

---

Below are the examples of Insight Agent Questions
"""
        followup_question = ["What is the total transport cost in 2024? (Insights Agent)",
                             "What is the trend of sales from July 2024 to Dec 2024? (Insights Agent)",
                             "List top 5 customers with highest pallet per order? (Insights Agent)",
                             "What is the trend of per pallet cost for ship to location ASDA? (Insights Agent)",
                             "Provide me some examples of Order Frequency Optimization Questions"]


    elif question == "Provide me some examples of Order Frequency Optimization Questions":

        description = """
Order Frequency Consolidation is designed to improve shipment planning by grouping customer orders, leading to better truck utilization and reduced logistics costs.

**Dynamic Order Frequency Consolidation** leverages a smart optimization algorithm that strategically combines customer orders within a flexible time window. This approach explores multiple consolidation possibilities and calculates potential cost savings, ultimately improving shipment efficiency and minimizing logistics expenses.

**Key Inputs Required:**
- **Customer Ship-to Location**  : Example- TESCO (BOLTON 510)  
  **or**  
  **Customer Postcode** : Example- NG17 1BX  
- **Shipment Window Range**: 2 to 10 days  
- **Truck Capacity**: Typically ranges between 26 to 52 pallets (e.g., 32 pallets)  
- **Utilization Threshold**: Between 60% to 95%

---

**Static Order Frequency Consolidation** follows a fixed dispatch schedule to group orders on specific days of the week. Unlike the dynamic approach, it doesn’t vary by shipment window but instead chooses the best fixed-day combinations to achieve high truck fill rates.

**Examples of Dispatch Scenarios:**
- 2-day: Shipments on Monday and Tuesday  
- 3-day: Shipments on Monday, Wednesday, and Friday  

The algorithm evaluates different weekly combinations to identify the most efficient shipping frequencies.

**Key Inputs Required:**
- **Customer Ship-to Location** : Example- TESCO (BOLTON 510)  
  **or**  
  **Customer Postcode** : Example- NG17 1BX  
- **Scenario Type** : Example- 1-day, 2-day, 3-day, etc.

---

Below are the examples of Order Frequency Optimization Questions
"""



        
        followup_question = ["What is the average shipment frequency per week for customer location BOOTS COMPANY PLC (SSC) in 2025? (Insights Agent)",
                             "What is the optimal shipment window for customer location ASDA if we want to keep utilization threshold 95% and consolidate orders for shipment window range (2–4)? (Dynamic Cost Optimization Agent)",
                             "What is the cost savings if we consolidate orders in 2024 for postcode 'CV' keeping capacity as 42 pallets in a 5 days window? (Dynamic Cost Optimization Agent)",
                             "What is the transport cost reduction for customer location BOOTS COMPANY PLC (SSC) in a 3-day delivery scenario keeping capacity of 40 pallets? (Static Cost Optimization Agent)",
                             "Provide me some examples of Drop Location Centralization Questions"]
        

    
    elif question == "Provide me some examples of Drop Location Centralization Questions":
        description = """
**Drop Point Centralization** is a strategic optimization algorithm designed to streamline distribution networks by reducing the number of delivery drop locations.The primary objective is to optimize delivery by minimizing unique drop postal codes for each customer and reducing total shipment distance, ultimately improving per-pallet cost efficiency and decreasing overall logistics spend.

**Key Inputs Required:**
- **Customer Sold-to location**  :Example: LIDL GB LIMITED 
- **Number of drop points**: Example: 1, 2, 3, etc. 
- **Ranking Parameter**: Rate, Distance, or Volume 

---

Below are the examples of Drop Point Centralization Questions
"""
        followup_question = ["What is the trend of cost per pallet for customer LIDL GB LIMITED in 2024? (Insights Agent)",
                             "For customer LIDL GB LIMITED, what is the cost per pallet if we centralize drop points? (Drop Point Centralization Agent)",
                             "For customer LIDL GB LIMITED, how does the transport cost changes if we optimize for 3 drop points ranked by Rate from 2024-07-01 to 2024-12-31? (Drop Point Centralization Optimization Agent)",
                             "Compare the cost savings if we optimize for 2 drop points vs 3 drop points for customer LIDL GB LIMITED (Drop Point Centralization Optimization Agent)",
                             "Provide me some examples of Pallet Utilization Questions"]
        
        
    
    elif question == "Provide me some examples of Pallet Utilization Questions":

        description = """
**Pallet Utilization Optimization** is an optimization algorithm focused on maximizing load efficiency by improving how products are arranged on pallets. It analyzes stacking patterns, layer configurations, and height adjustments—including the potential for double stacking—to increase pallet fill rates. The goal is to reduce unused space, lower the number of pallet required, and ultimately drive down transportation costs and overall logistics spend.

**Key Inputs Required:**
- **Material**  :Example: 5000033048
- **Pallet Type**: Industrial, Euro or Euro CCG1. 
- **Double Stacking on Storage**: True or False
- **Inbound Ocean Freight**: True or False

---

Below are the examples of Pallet Utilization Optimization Questions
"""
        followup_question = ["List 5 materials with lowest layer utilization % (Insights Agent)",
                             "What is the new layer utilization % for material 5000033048 after optimizing layer configuration? (Pallet Utilization Optimization Agent)",
                             "For material 5000033048 on EURO pallets what change in pallet required we can get by optimizing layer and adjusting height? (Pallet Utilization Optimization Agent)",
                             "What is the cost impact if height is adjusted to 1.85 for material 5000033048 and inbound is through ocean freight? (Pallet Utilization Optimization Agent)",
                             "Provide me some examples of Insight Questions"]
    

    elif question == "Please tell about your capabilities and datasets":

        description = """ I can assist with various tasks related to shipment and cost optimization, including analyzing shipment data, generating insights, optimizing costs through different strategies, and creating scenario analyses. The datasets I work with include shipment data, SKU master data, and cost-related information. If you have specific questions about what I can do or the types of analyses I can perform, feel free to ask!"""
        followup_question = ['Brief me about key agents and optimisation strategies available',
                                 'Provide me some examples of Insight Questions',
                                 'Provide me some examples of Order Frequency Optimization Questions',
                                 'Provide me some examples of Drop Location Centralization Questions',
                                 'Provide me some examples of Pallet Utilization Questions']
    
    elif question == "Brief me about key agents and optimisation strategies available":
        description = """
## Key Agents Available

- **Insights Agent**  
  Analyzes shipment and SKU master data to generate insights, perform exploratory data analysis, and identify trends.

- **Order Frequency Optimization Agent**  
  Optimizes shipment planning by consolidating customer orders to improve delivery efficiency and reduce logistics costs.  
  It includes two complementary approaches:
  - **Dynamic Consolidation**: Analyzes shipment patterns and dynamically groups orders across time windows to optimize cost and utilization.
  - **Static Consolidation**: Assesses fixed-day delivery schedules to reduce shipment costs by grouping deliveries on selected days of the week.

- **Pallet Utilization Optimization Agent**  
  Identifies cost savings by optimizing pallet utilization through better use of pallet specifications.

- **Drop Point Centralization Optimization Agent**  
  Optimizes the number and location of delivery drop points to reduce transport costs and CO₂ emissions.

These agents utilize different strategies to enhance efficiency, reduce costs, and improve overall logistics performance.
"""
        followup_question = ['Please tell about your capabilities and datasets',
                             'Provide me some examples of Insight Questions',
                             'Provide me some examples of Order Frequency Optimization Questions',
                             'Provide me some examples of Drop Location Centralization Questions',
                             'Provide me some examples of Pallet Utilization Questions']

    return description, followup_question 


