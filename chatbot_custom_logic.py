import pyodbc
import pandas as pd
import numpy as np
import redis
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import warnings
import time
import json
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

fee_memory = ConversationBufferWindowMemory(k=7)
fee_chain = ConversationChain(
    llm=llm_claude,
    memory=fee_memory
)

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

def get_column_defination(columns):
    column_keys = list(column_defination.keys())
    description = ""
    for column in columns:
        for col in column_keys:
            if column.lower() == col.lower():
                description += f'- {column}: {column_defination[col]}\n'
    return description

def process_user_query(user_query, mandatory_fields, optional_fields):
    # Extract column names for mandatory and optional fields
    mandatory_columns = [field["column_name"] for field in mandatory_fields]
    optional_columns = [field["column_name"] for field in optional_fields]


    # metadata = fee_metadata
    conversation_history = fee_memory.buffer
    # Create a unified prompt
    prompt = f"""
    You are an intelligent assistant. Analyze the following user query and determine its intent and memory context:
    
    Possible intents:
    1. "fee_change" - If the user query explicitly want to change or modify the fee schedule, rate, factor then and only then intent is 'fee_change' otherwise set the default intent.
    2. "query_database" - This is the default intent if the intent is not 'fee_change'.

    User Query: "{user_query}"
    Conversation_history: "{conversation_history}"

    Instructions:
    1. Determine the intent of the query based on the user input and conversation history.
    2. Check if the mandatory and optional details are available in conversational memory:
       - Mandatory Details: {', '.join(mandatory_columns)}
       - Optional Details: {', '.join(optional_columns)}
    3. If any detail is missing in User Query then must get it from the conversation history
    3. If details are missing, identify which ones and propose a clarification message to the user.
    4. In the clarification message include the default values for missinng details and confirm if user \
    wants to use the default values
    5. default values are 'StartDate' = '1st jan 2018', 'end_date' = '31st Dec 2020'
    6. If the query is ambiguous, propose clarification questions based on the metadata and user input.

    Return only JSON object with the following structure :
    {{
        "intent": "determined_intent",
        "message": "Dynamic response message based on memory analysis.",
        "can_run": 0 or 1,
        "mandatory_available": 0 or 1,
        "optional_available": 0 or 1,
        "available_mandatory": {{"column_name": "value", ...}},
        "available_optional": {{"column_name": "value", ...}},
        "optional_denied": 0 or 1
    }}
    """

    try:
        response = llm_claude.invoke(input = prompt).content
        # print("Fee processing", response)
        # Parse the JSON response
        response_data = json.loads(response)
        return response_data
    except json.JSONDecodeError as json_err:
        print(f"JSON parsing error: {json_err}")
        print(f"Raw LLM Response: {response}")
        return {
            "intent": "other",
            "message": "An error occurred while analyzing your query.",
            "can_run": 0,
            "mandatory_available": 0,
            "optional_available": 0,
            "available_mandatory": {},
            "available_optional": {},
            "optional_denied": 0
        }
    except Exception as e:
        print(f"Error processing user query: {e}")
        return {
            "intent": "other",
            "message": "An error occurred while analyzing your query.",
            "can_run": 0,
            "mandatory_available": 0,
            "optional_available": 0,
            "available_mandatory": {},
            "available_optional": {},
            "optional_denied": 0
        }


connectionString = f'DRIVER={{FreeTDS}};SERVER={SERVER};DATABASE={DATABASE};UID={USERNAME};PWD={PASSWORD}'

def can_run(parameters):
    account_number = parameters.get('AccountNumber')
    start_date = parameters.get('StartDate')
    end_date = parameters.get('EndDate')
    fee_schedule = parameters.get('FeeSchedule')
    factor = parameters.get('Factor')
    
    # Check if AccountNumber, StartDate, and EndDate are present
    if account_number and start_date and end_date:
        # Check if either FeeSchedule or Factor (or both) are present
        if fee_schedule or factor:
            return True
    
    return False

def get_fee_schedule_impact(parameters):
    account_number = parameters.get('AccountNumber')
    start_date = parameters.get('StartDate')
    end_date = parameters.get('EndDate')
    fee_schedule = parameters.get('FeeSchedule')
    factor = parameters.get('Factor')
    query = f"SELECT * FROM HVT_Fee_Details WHERE CompositeAccount_IDFK = {account_number} and Date >= '{start_date}' and date <= '{end_date}'"
    
    with pyodbc.connect(connectionString) as conn:
        with conn.cursor() as cursor:
            result = pd.read_sql(query, conn)
    if result.empty:
        return []
    else:
        result = result.head(50)[['CompositeAccount_IDFK', 'Date', 'Name', 'Type', 'MarketValue', 'factor', 'FeeSchedule', 'Total_Fees']]
        old_details = result.to_dict(orient='records')
        if fee_schedule:
            result['FeeSchedule'] = float(fee_schedule)
        if factor:
            result['factor'] = float(factor)
        result['Total_Fees'] = result['MarketValue'] * result['FeeSchedule'] * result['factor'] * 0.01
        new_details = result.to_dict(orient='records')
        return [{'old_detials':old_details, 'new_details': new_details}]

description  = get_column_defination(columns)
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


def main(user_query):
    # Define mandatory and optional columns
    mandatory_columns = [
        {"column_name": "AccountNumber", "column_desc": "It is the account number for which fees need to be calculated."},
        {"column_name": "StartDate", "column_desc": "The account portfolio's market value changes on this date, which serves as the starting point for fee calculation."},
        {"column_name": "EndDate", "column_desc": "The account portfolio's market value changes on this date, which serves as the ending point for fee calculation."},
        {"column_name": "FeeSchedule", "column_desc": "This is the rate which is applied to calculate the fees"},
        {"column_name": "Factor", "column_desc": "This is the factor multiplied to calculate the fees "},
    ]
    
    optional_columns = [
        {"column_name": "class", "column_desc": "the travel class (e.g., economy, business)"},
        {"column_name": "vendor", "column_desc": "The name of the vendor or supplier involved in the transaction, such as a hotel, airline, or service provider"},
    ]
    
    outcome = process_user_query(user_query, mandatory_columns, optional_columns)
    intent = outcome['intent']
    fee_parameters = outcome['available_mandatory']
    response = None
    print(outcome)
    if intent == 'query_database':
        response = frame_fee_response(user_query)
    if intent == 'fee_change':
        fee_memory.chat_memory.add_user_message(user_query)
        fee_memory.chat_memory.add_ai_message(f'{fee_parameters}')
        if can_run(fee_parameters):
            results = get_fee_schedule_impact(fee_parameters)
            print(results)
            response = generate_fee_summary_v2(user_query, results)
    print(response)
