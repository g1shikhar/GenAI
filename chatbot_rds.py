# Importing dependencies

import pyodbc
import pandas as pd
import numpy as np
import redis
from langchain_redis import RedisConfig, RedisVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import warnings
import time
import json
from redisvl.query.filter import Tag
from langchain_aws import ChatBedrock
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain_aws import ChatBedrock
import re
import boto3
from datetime import datetime
from decimal import Decimal
import time
from langchain_openai import ChatOpenAI

pd.set_option("display.max_columns",100)
# Ignore all warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
BEDROCK_MODEL_ID = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"\


# Script to configure ODBC connector in jupyter NB, foir detail ask gemini to explain the below code
%%bash
sudo apt-get update
sudo apt-get install -y unixodbc unixodbc-dev freetds-dev freetds-bin tdsodbc\

%%writefile /tmp/odbcinst.ini
[FreeTDS]
Description = FreeTDS Driver
Driver = /usr/lib/x86_64-linux-gnu/odbc/libtdsodbc.so
Setup = /usr/lib/x86_64-linux-gnu/odbc/libtdsS.so
!sudo cp /tmp/odbcinst.ini /etc/odbcinst.ini


# Getting AWS credential
secret_manager_client = boto3.client('secretsmanager',region_name='ap-south-1')
redis_url = json.loads(secret_manager_client.get_secret_value(SecretId='your_url_key')['SecretString'])['REDIS_URL']
rds = json.loads(secret_manager_client.get_secret_value(SecretId='your_credential_json')['SecretString'])
rds_username = rds['username']
rds_password = rds['password']
rds_engine = rds['engine']
rds_host = rds['host']
rds_port = rds['port']
rds_dbInstanceIdentifier = rds['dbInstanceIdentifier']
SERVER = f'{rds_host},{rds_port}'
DATABASE = 'DB_name_to_connect_inaws_rds'
USERNAME = rds_username
PASSWORD = rds_password

#Initiating LLM from bedrock
BEDROCK_MODEL_ID = "your aws llm model id"
def bedrock_llm(model_id=BEDROCK_MODEL_ID):
    llm = ChatBedrock(
        model_id=model_id,
        temperature=0,
        region_name="us-east-1"
    )
    return llm

llm_claude = bedrock_llm()


# OPENAI LLM
OPENAI_API_KEY="you open api key"
def get_openai_response():
    """
    Function to interact with OpenAI models based on the provided model name.
    Catches errors and allows fallback to other models.
    """
    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-4o",
    )
    return llm

llm_openai = get_openai_response()


# Configuring LLM memory to store context inside langchain
memory = ConversationBufferWindowMemory(k=7)
chain = ConversationChain(
    llm=llm_openai,
    memory=memory
)

# Configuring aws claudia LLM memory to store context inside langchain
fee_memory = ConversationBufferWindowMemory(k=7)
fee_chain = ConversationChain(
    llm=llm_claude,
    memory=fee_memory
)

#defining redis index for storing metedata
redis_config_v2 = RedisConfig(
    index_name="mpnet_v16",
    redis_url=redis_url,  # Update Redis URL
    metadata_schema=[
        {"name": "table_name", "type": "tag"},
        {"name": "column_name", "type": "text"},
        {"name": "column_desc", "type": "text"}
    ],
    embedding_field="embedding_v16",
    embedding_field_dtype="float32"
)

metadata_for_rds_tbl =[
    {'table_name': 'rds_table1',
     "description": [ "description of rds_table1"],
     'column_names': ['col1', 'col2', 'col3', 'col4'], 
     'column_details': [{'column_name': 'col1', 'column_desc': 'col1 desc'}, 
                        {'column_name': 'col2', 'column_desc': 'col2 desc'}, 
                        {'column_name': 'col3', 'column_desc': 'col3 desc'}, 
                        {'column_name': 'col4', 'column_desc': 'col4 desc}]}, 
                         
    {'table_name': 'rds_table2',
     "description": ["description of rds_table1"],
     'column_names': ['col1', 'col2', 'col3', 'col4', 'col5'],
     'column_details': [{'column_name': 'col1', 'column_desc': 'col1 desc'}, 
                        {'column_name': 'col2', 'column_desc': 'col2 desc'}, 
                        {'column_name': 'col3', 'column_desc': 'col3 desc'}, 
                        {'column_name': 'col4', 'column_desc': 'col4 desc},
                        {'column_name': 'col5', 'column_desc': 'col5 desc}]},
]



# function to retrieve table name from VectorStore
def query_main_index(query_text: str, k: int):
    # Initialize embeddings model and RedisVectorStore for the main index
    hf_embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    vector_store_main = RedisVectorStore(embeddings=hf_embeddings, config=redis_config_v2)
    
    # Perform similarity search to find relevant table
    table_results = vector_store_main.similarity_search(query_text, k=k)
    # print(table_results)

    table_names = []
    for doc in table_results:
        table = {}
        table['table_name'] = doc.metadata['table_name']
        table['column_names'] = doc.metadata['columns']
        table_names.append(table)
    return table_names


# function to retrieve column names from table
def query_columns_for_table(table_name: str, query_text: str):
    """
    Queries the Redis vector store to retrieve column details for a specific table based on a given query.

    This function uses an embeddings model to perform similarity searches on a vector store. 
    It filters results to only include columns associated with the specified table name and retrieves metadata 
    such as column names and descriptions.

    Args:
        table_name (str): The name of the table to filter the column search.
        query_text (str): The query string to match columns based on similarity.

    Returns:
        tuple: A tuple containing:
            - columns (list): A list of column names that match the query.
            - column_details (list): A list of dictionaries, where each dictionary contains:
                - 'column_name': The name of the column.
                - 'column_desc': The description of the column.
              Returns an empty list and `None` if an exception occurs.

    """
    try:
        # Initialize the HuggingFace embeddings model
        hf_embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        
        # Initialize the Redis vector store with the embeddings model and configuration
        vector_store_v2 = RedisVectorStore(embeddings=hf_embeddings, config=redis_config_v2)
        
        # Define the filter condition to match the specified table name
        filter_condition = Tag("table_name") == table_name
        
        # Perform a similarity search with the query text and filter condition
        column_results = vector_store_v2.similarity_search(query_text, filter=filter_condition, k=15)
        
        # Initialize lists to store column names and details
        columns = []
        column_details = []
        
        # Extract column metadata from the search results
        for column_doc in column_results:
            column_detail = {
                'column_name': column_doc.metadata.get('column_name'),
                'column_desc': column_doc.metadata.get('column_desc')
            }
            columns.append(column_doc.metadata.get('column_name'))
            column_details.append(column_detail)
        
        return columns, column_details

    except Exception as e:
        return [], None


def table_metadata(query, k):
    """
    Retrieves metadata for tables based on a query, including column names and details.

    This function first queries the main index to get a list of table names matching the query. 
    For each table, it fetches associated column names and their details using the 
    `query_columns_for_table` function. The result is a list of dictionaries containing 
    table names, column names, and column details.

    Args:
        query (str): The query string to match tables and their metadata.
        k (int): The number of top matching tables to retrieve.

    Returns:
        list: A list of dictionaries where each dictionary contains:
            - 'table_name': The name of the table.
            - 'column_names': A list of column names associated with the table.
            - 'column_details': A list of dictionaries, each containing:
                - 'column_name': The name of the column.
                - 'column_desc': The description of the column.
    """
    result = []  # Initialize the result list to store table metadata

    # Query the main index to get a list of table names matching the query
    table_names = query_main_index(query, k)
    
    # Iterate over each table retrieved from the main index
    for table in table_names:
        print(table)  # Debugging: Print the table information
        
        # Fetch columns and column details for the table
        columns, column_details = query_columns_for_table(table['table_name'], query)
        print("columns", columns)  # Debugging: Print the retrieved columns
        
        # If columns exist, add them to the table dictionary
        if columns:
            table['column_names'] = columns
            table['column_details'] = column_details
        
        # Append the updated table dictionary to the result list
        result.append(table)
    
    return result





def execute_query(sql_query):
    """
    Executes a SQL query on a specified database, cleans the results, and returns processed data.

    This function connects to a database using `pyodbc`, executes the provided SQL query, and retrieves 
    the result set. It processes the results to handle specific data types (e.g., `datetime`, `Decimal`), 
    removes rows with zero values in specified columns, and returns the final cleaned data along with column names.

    Args:
        sql_query (str): The SQL query to be executed.

    Returns:
        tuple: A tuple containing:
            - final_result (list): A list of dictionaries where each dictionary represents a row of the processed data.
            - columns (list): A list of column names from the query result.
    """
    # Construct the connection string for the database
    connectionString = f'DRIVER={{FreeTDS}};SERVER={SERVER};DATABASE={DATABASE};UID={USERNAME};PWD={PASSWORD}'
    
    with pyodbc.connect(connectionString) as conn:
        with conn.cursor() as cursor:
            # Execute the SQL query and fetch all results
            result = cursor.execute(sql_query).fetchall()
            # Retrieve column names from the query result
            columns = [column[0] for column in cursor.description]

    # Process and clean the result set
    clean_result = [
        {
            column: (
                value.strftime("%m-%d-%Y") if isinstance(value, datetime)  # Format datetime values
                else round(float(value), 2) if isinstance(value, Decimal)  # Round Decimal values
                else value  # Keep other values as is
            )
            for column, value in zip([desc[0] for desc in cursor.description], row)
        }
        for row in result
    ]
    
    # Convert the cleaned result set into a Pandas DataFrame
    df = pd.DataFrame(clean_result)
    
    # Columns to check for zero values
    columns_to_clean = ['TickerSymbol', 'InstrumentIDFK', 'CompositeAccountIDFK', 'CompositeAccount_Parent_IDFK', 'ParentIDFK']
    
    # Fill all NaN values with 0
    df.fillna(0, inplace=True)
    
    # Remove rows with zero values in the specified columns
    for column in df.columns:
        if column in columns_to_clean:
            df = df[df[column] != 0]
    
    # Convert the final DataFrame back into a list of dictionaries
    final_result = df.to_dict(orient='records')
    
    return final_result, columns

def check_user_query_v2(query, llm):
    # Formulate the prompt for the language model
    prompt = f"""
        You are an expert in English language. You are given an input question. \
        Given an input question and below information, you need to set the flag as 1 or 0 and also return a default_message using below information and Instructions. \ 
        Follow the Information:
        - The table SampleSurveillanceData keeps a record of Surveillance Data for various stocks or financial instrument under surveillance. \
          It helps to determine whether a ticker symbol for a financial instrument is under surveillance and the date when it was put under surveillance.
        - The table HVT_HoldingsAveraged keeps a record of averaged holdings data for various financial instruments over specific periods. \
          Each row represents detailed information on a particular holding, including metrics like quantity, market value, and accruals. \
          The data is structured to provide aggregated values for easier analysis, helping to assess asset performance over time. \
          It enables the tracking of investment values on given dates, which is essential for generating accurate financial reports \
          and making informed investment decisions. \
        - The table HVT_Instrument provides a list of financial instruments available for investment, capturing key details about each instrument. \
          Each row represents an individual instrument, covering its name, description, and current status to support investment tracking. \
          The data helps users monitor instruments' characteristics, including cash or fund classifications, for efficient portfolio management. \
          By maintaining details on active status, this table enables users to make informed decisions regarding their investment options. \
        - The table HVT_InstrumentSymbol keeps track of the symbols used to identify each financial instrument. Think of it as a central place for \
          storing ticker symbols or unique codes that represent each asset. \
          Each record includes details about the symbol name, the actual value or ticker, and whether it's the main symbol used for that instrument. \
          This makes it easier for people working in trading or financial analysis to quickly find and reference specific instruments. \
          By standardizing these symbols, the table also helps make sure that different systems (like trading platforms) recognize the same assets in a consistent way \
        - The table HVT_CompositeAccount represents the structure for storing composite account details. \
          Each record captures essential attributes of a user's composite accounts, including account types, account names, and identifiers. \
          Tracks account creation, updates, and import details for effective account management. \
          Facilitates the organization and classification of accounts for improved financial tracking and portfolio management. \
          It also can be used to get user details such as account name, account number, etc.
        - The table HVT_CompositeAccount_CompositeAccount_Mapping Establishes hierarchical relationships between \
          composite accounts and their parent accounts. \
          Each record links a composite account to its parent, enabling structured account management. \
          Tracks creation and update information to maintain data integrity. \
          Facilitates navigation and organization of accounts within the financial system. \
         -The table HVT_Ticker_name_symbol_asset_sector contains information about various financial instruments, including their ticker symbols,  account fee, fee details, ticker names, their asset classes, sectors and market capitalization categories. \
          It can be used to identify financial assets, asset class based on their ticker symbols and retrieve details like their industry sectors and market cap classifications. \
          The table is suitable for applications related to financial analysis, portfolio management, and sector-based asset classification. \
         - The table HVT_Fee_Details contains the details related to the fee, fee, details, fee schedules, fee type.
        Follow these Instructions for setting the 'default_message' and 'flag' value: \
         - If the question is related to above provided Information then set the flag as 1 \
           else set the flag as 0.
         - If the flag is set to 0 then rephrase the input question and ask whether he is enquiring about account number, \
           account portfolio, account market value, account fee and fee details etc. depending upon the user query, then return rephrased question as default_message and flag. \
         - If the flag is set to 1 then simply return input question as default_message and flag. \
        - Only return the JSON with keys 'flag' and 'default_message' with their values and nothing else. \
        question: {query}
    """
    
    # Invoke the language model chain with the generated prompt
    output = llm.invoke(input=prompt).content
    # print("o/p: ",output)
    
    # Extract the response from the output and process it
    response = re.sub(r'\s{2,}|\\n', ' ', output)
    response = json.loads(re.sub(r"```json|```", '', response))
    print("output", response)
    
    # Retrieve the extracted 'account_id' and 'default_message'
    flag = response.get("flag")
    default_message = response.get("default_message")
    
    # Return the extracted values
    return flag, default_message


def check_query_for_fee(query, llm):
    # Formulate the prompt for the language model
    prompt = f"""
        You are an expert in English language. You are given an input question. \
        Given an input question and below information, you need to set the flag as 1 or 0 and also return a default_message using below information and Instructions. \ 
        Follow the Information:
        
        -The table HVT_CompositeAccount_BillingScheme_Mapping defines the relationship between composite accounts \
         and their associated billing schemes.\
         Each record specifies the mapping of composite accounts to billing schemes for accurate financial processing.\
        -The table HVT_BillingScheme_BillingSchemeSubSchedule_Mapping maps billing schemes to their respective sub-schedules,\
         establishing relationships between different billing components.\
         Each record links a billing scheme identified by 'BillingSchemeIDFK' to a specific sub-schedule through 'BillingSchemeSubScheduleIDFK'.\
         It helps in facilitating the management and retrieval of billing information based on specific scheduling requirements.\
        -The table HVT_BillingSchemeSubSchedule contains detailed information about billing scheme sub-schedules.\        
         It tracks important attributes like 'Type', 'ScheduleStyle', and whether the schedule is currently 'IsActive'.\
         It Allows specification of financial parameters including 'MinimumFee', 'MaximumFee', and flags for including uninvested cash and accrual.",
         Also it indicates whether charges are annual and whether rates are expressed as percentages.\            
        -The table HVT_BillingSubScheme_BillingBrackets_Mapping maps billing sub-schemes to their corresponding billing brackets.\
         Each record includes identifiers for the billing scheme sub-schedule and the associated billing bracket.\
         It tracks the 'IsActive' status to indicate whether the mapping is currently in use.\
         Facilitates the organization of billing structures by linking various billing components together.\                  
        -The table HVT_BillingBrackets defines the billing brackets used for calculating fees based on specified ranges.\
         Each entry includes a low and high value representing the range for which the rate applies.\
         It allows flexibility in pricing models by specifying rates either as a percentage or as a fixed value.\
         It helps in determining charges based on customer usage.\
        -The table HVT_CompositeAccount_FeeAdjustment_Mapping tracks fee adjustment mappings for composite accounts.\
         It includes foreign keys FeeAdjustmentIDFK, FeeTypeIDFK to link composite accounts, fee adjustments,\
         and fee types, ensuring structured and traceable financial adjustments.\
        -The table HVT_FeeAdjustments records fee adjustments applied to accounts, detailing the type and value of each adjustment.\
         Includes data about the adjustment, such as start and end dates, to track validity periods.\                
         It allows for categorization of adjustments for reporting and analysis purposes.\
        -The table HVT_fee_details is dedicated to fee calculation and contains all parameters, thresholds, and
         adjustment values needed to compute fees.\
         It captures comprehensive details such as billing schemes, market values, adjustment types,\
         and calculated fees to ensure accurate financial management.\
         The data in this table supports generating detailed insights, optimizing billing structures, and enabling precise fee computations.\  
         
        Follow these Instructions for setting the 'default_message' and 'flag' value: \
         - If the question is related to above provided Information then set the flag as 1 \
           else set the flag as 0.
         - If the flag is set to 1 then rephrase the input question and ask whether he is enquiring about fee details. \
             depending upon the user query, then return rephrased question as default_message and flag. \
         - If the flag is set to 0 then simply return input question as default_message and flag. \
        - Only return the JSON with keys 'flag' and 'default_message' with their values and nothing else. \
        question: {query}
    """
    
    # Invoke the language model chain with the generated prompt
    output = llm.invoke(input=prompt).content
    # print("o/p: ",output)
    
    # Extract the response from the output and process it
    response = re.sub(r'\s{2,}|\\n', ' ', output)
    response = json.loads(re.sub(r"```json|```", '', response))
    print("output", response)
    
    # Retrieve the extracted 'account_id' and 'default_message'
    flag = response.get("flag")
    default_message = response.get("default_message")
    
    # Return the extracted values
    return flag, default_message



def get_surveillancedata_response(query, memory, custom_metadata=[]):
    """
    Generates and executes an MSSQL query based on a natural language input and table metadata.

    Args:
        query (str): The natural language question or query from the user.
        chain: The language model or chain used to generate the MSSQL query.

    Returns:
        tuple: A tuple containing:
            - results (list of dict): The query results, where each row is represented as a dictionary.
            - col_names (list): List of column names in the result set.
            - generated_query (str): The MSSQL query generated by the chain.
        If an error occurs, returns (None, None, None).

    Raises:
        Exception: If query generation or execution fails.
    """
    metadata = custom_metadata
    conversation_history = memory.buffer
    # Prompt to generate the MSSQL query
    join_prompt = f"""
        You are an MSSQL expert. You are given an input question and context and Conversation_history. \
        Conversation_history is the record of all previous interactions that occurred with you previously. \
        Also, refer to conversation_history if the query is related to past conversation. \
        The context contains and array of dictionaty with keys as \
        table_name which is the name of the SQL Table, description which is the description of the SQL Table \
        , column_names which has column names of the SQL table and column_details which is a list of dictionary \
        with keys as column_name and column_desc which provide description of the column_name.
        Given an input question, use the information in the context to create a syntactically correct MSSQL query. \
        Follow these instructions for creating a syntactically correct SQL query: \
        - Context contains more than one table, so you can create a query by performing JOIN operations if required \
        - Pay close attention to which column belongs to which table. \
        - Use the 'column_desc' key under column_details in the context to choose the right columns based on user input query. \
        - Do not query for columns that do not exist in the tables. \
        - Always add StockSurveillanceStatus column in select clause of MSSQL query if the input question is related to Surveillance. \
        - Use aliases only where required. \
        - Must consider that ORDER BY items must appear in the select list if SELECT DISTINCT is specified.
        - For operations such as average (AVG) or ratio, use the appropriate aggregation function. \
        - Incorporate filtering criteria mentioned in the question using the WHERE clause. \
        - Use logical operators (AND, OR) for combining multiple conditions. \
        - For date or timestamp columns, use appropriate date functions (e.g., DATE_PART, EXTRACT). \
        - If grouping data is required, use the GROUP BY clause with aggregate functions. \
        - Use single-letter aliases for tables and columns to improve query readability. \
        - If an alias is then used then use the same alias everywhere in the query and not the original name \
        - Avoid using SQL commands or keywords as aliases. \
        - Use subqueries or CTEs if necessary for complex queries. \
        - Convert any Date in the recommendation summary to us-date format.
        - Fetch the most recent results only if it is feasible. If retrieving the latest information is possible, proceed to do so; otherwise, no action is necessary.
        - Convert any amount in the recommendation summary to  us-number format such as "$6,000,999.0". 
        - Then only return the mssql query in string and nothing else.

        Question: {query}
        Context: {metadata}
        Conversation_history: {conversation_history}
    """
    try:
        output = llm_openai.invoke(input=join_prompt)
        response = output.content
        generated_query = response
        print(generated_query)
        generated_query = re.sub(r"```sql|```", '', generated_query)
        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(response)
        try:
            results, col_names = execute_query(generated_query)
            message=""
            if len(results)>100:
                results=results[:100]
                message = "Giving recommendation for 100 rows only. please create aggregate queries."
            # Raise an exception if no results are returned
            if not results:
                raise Exception("No results returned from the query execution.")
            # memory.chat_memory.add_user_message(f"The SQL response generated for the MSSQL query {generated_query} is {results}")
            memory.chat_memory.add_user_message(f"The results fetched from the DB for the query {generated_query} is {results}")
            memory.chat_memory.messages = memory.chat_memory.messages[-8:]
            return results, message
        
        except Exception as inner_e:
            print(f"Error executing join prompt query: {inner_e}\n")
            # Remove the last inserted message if an error occurs
            if len(memory.chat_memory.messages)>2:
                memory.chat_memory.messages.pop()
                memory.chat_memory.messages.pop()
            raise

    except Exception as outer_e:
        print(f"Error generating SQL query: {outer_e}\n")
        raise
        return None, None



column_defination = {
    "CompositeAccount_IDFK": "A foreign key that links to the child composite account in the account hierarchy. This identifies the account that is a child of a parent composite account.",
    "CompositeAccount_Parent_IDFK": "A foreign key that links to the parent composite account in the hierarchy. This identifies the higher-level composite account that a child account is associated with.",
    "ID": "A unique identifier for each mapping record, ensuring that each relationship between composite accounts and billing schemes can be individually referenced.",
    "IsActive": "A boolean value indicating whether the mapping is currently active. If true, the mapping is in use; if false, it is inactive and not currently relevant.",
    "CompositeAccountIDFK": "A foreign key that links to the Composite Account table. It identifies the specific composite account associated with the billing scheme in this mapping.",
    "BillingSchemeIDFK": "A foreign key that links to the Billing Scheme table. It identifies the billing scheme associated with the composite account in this mapping.",
    "BillingSchemeSubScheduleIDFK": "Foreign key referencing the billing scheme sub-schedule, specifying the sub-schedule associated with the billing scheme.",
    "Name": "The full, official name of the billing sub-schedule, which provides a clear description of its purpose. This is also the name or title of the fee adjustment, explaining what it's for, like a label or heading.",
    "ShortName": "A brief, easy-to-reference name or abbreviation for the billing sub-schedule. This is also a shorter version of the name, a quick reference or nickname for the fee adjustment.",
    "Description": "A simple, short explanation of what the billing sub-schedule is and how it works. This also provides a detailed explanation of what the fee adjustment is, the full story behind the adjustment.",
    "Type": "The category or classification of the billing sub-schedule, indicating its purpose or usage.",
    "ScheduleStyle": "The format or structure of the billing schedule, describing how charges are applied or calculated.",
    "IsActive": "Shows whether this billing sub-schedule is currently in use or has been disabled.",
    "MinimumFee": "The minimum charge that can be applied for this billing sub-schedule, regardless of other factors.",
    "MaximumFee": "The highest charge that can be applied for this billing sub-schedule, limiting the fee amount.",
    "IncludeUninvestedCash": "Indicates whether any uninvested cash should be considered when calculating charges in this sub-schedule.",
    "IncludeAccrual": "Shows if any accumulated amounts or unpaid charges are included in this billing sub-schedule.",
    "IsAnnualCharges": "Indicates whether the charges are applied on an annual basis or over a different period.",
    "ExternalID": "An external reference number or identifier used to connect this sub-schedule to other systems or processes.", 
    "FeeTypeIDFK": "References the type of fee applied in this sub-schedule, such as a flat fee or percentage-based fee.",
    "MarketValueLimit": "Sets a maximum limit based on the market value for which this sub-schedule is applied, if applicable.",
    "IsRatePercent": "Indicates if the charge is based on a percentage of some value (e.g., market value, revenue, etc.).",
    "IsCriteriaRatePercent": "Shows if the rate depends on certain conditions being met to apply a percentage fee.",
    "BillingBracketsIDFK": "A foreign key reference to the billing bracket being used in this mapping, connecting it with a specific billing sub-scheme.",
    "LowValue": "Represents the minimum value in the billing range, defining the lower bound for the applicable fee or charge.",
    "HighValue": "Represents the maximum value in the billing range, defining the upper bound for the applicable fee or charge.",
    "RatePercentOrValue": "Indicates the rate to be applied within the specified range, either as a percentage or a fixed value for the billing calculation.",
    "IsThreshold": "A flag that determines whether the billing bracket represents a threshold for special pricing considerations, if applicable.",
    "CompositeAccountIDFK": "A foreign key linking to the composite account that the fee adjustment is applied to, identifying which account is being affected by the adjustment.",
    "FeeAdjustmentIDFK": "A foreign key linking to the specific fee adjustment record that details the financial change applied to the composite account.",
    "FeeTypeIDFK": "A foreign key linking to the type of fee adjustment, defining the category or purpose of the fee, crucial for fee management.",
    "FeeAdjustmentType": "This is a number representing the type of adjustment, like whether it's a discount or penalty.",
    "StartDate": "This is when the fee adjustment starts being effective, like saying 'From this date onwards, this adjustment applies. The starting date of the billing cycle or fee applicability, used in temporal queries.",
    "EndDate": "This shows when the fee adjustment stops being effective, if there's no end date, it means the adjustment is ongoing. The ending date of the billing cycle or fee applicability, aiding date range-based queries.",
    "Value": "This is the amount of the fee adjustment, it could be a positive number (credit/discount) or negative (penalty/charge).",
    "IsPercent": "This tells you if the fee adjustment is a percentage (true) or a fixed amount (false), like 10% or $50.",
    "LowValue": "The minimum threshold for the payout bracket, indicating the starting point of the payout range.",
    "HighValue": "The maximum threshold for the payout bracket, indicating the upper limit of the payout range.",
    "RatePercentOrValue": "The rate applied within this payout bracket, which can be a percentage or a fixed value for payout calculations.",
    "Factor": "Multiplier or factor applied in calculations, significant for financial models.",
    "FeeSchedule": "Represents the fee schedule applied on account AUM to calculate the final fee.",
    "Total_Fees": "The final calculated fee, serving as a result metric in LLM-generated reports.",
}


# getting description of only for the columns which are there in results fetched from rds (get_surveillancedata_response)
def get_column_defination(columns):
    column_keys = list(column_defination.keys())
    description = ""
    for column in columns:
        for col in column_keys:
            if column.lower() == col.lower():
                description += f'- {column}: {column_defination[col]}\n'
    return description


# function to generate summary for the results fetched from rds (get_surveillancedata_response) 
def generate_fee_summary_v2(query, data):
    description = ""
    if data:
        columns = list(data[0].keys())
        description  = get_column_defination(columns)
    fee_summary_prompt = f"""
    You are a summary generator. You are provided a question and context as input.
    You have to generate a descriptive summary as per the question for the provided context.
    Context is given in the form of a single list of dictionaries.
    The context is the results from the Database and correct as per the questions.
    Information about columns is as follows:
    {description}
    Total_Fees:  This is the final calculated fee for account number.
    The formula for fee calculation is 'MarketValue'*'calculation_schedule'*'factor'*0.01 \
    where MarketValue, calculation_scheddule, and factor are defined above.
    If the question asks for impact, comparison, and similar requirement then use the formula for fee calculation to develop new fees. \
    Get the required values from context.
    question: {query}
    context: {data}
    """
    output = llm_claude.invoke(input=fee_summary_prompt)
    summary = output.content
    return summary


def frame_fee_response(query):
    results=[]
    try:
        flag, default_message = check_user_query_v2(query, llm_claude)
        flag2, default_message2 = check_query_for_fee(query, llm_claude)
        
        if flag==1 and flag2==1:
            print(default_message2)
            results, message = get_surveillancedata_response(query, fee_memory ,metadata_for_rds_tbl)
    except:
        pass
    print(results)
    if not results:
        fee_summary = "Sorry not able to generate results for above query"
    else:
        fee_summary = generate_fee_summary_v2(query, results)
    print(fee_summary)
