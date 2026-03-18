"""LangGraph Workflow — the core extraction pipeline.

Nodes:
  text_extract → scraper → retrieval → llm_extract → enrichment → evaluate
                                                                      │
                                                            fill_rate < threshold?
                                                            ┌─────┴─────┐
                                                            ▼           ▼
                                                       web_fallback    END
                                                            │
                                                            ▼
                                                       re_evaluate
                                                            │
                                                            ▼
                                                           END

Usage:
    from agent_server.sec_extraction.workflow import extraction_workflow
    result = extraction_workflow("Your SEC filing text...")
    print(result["record"])
    print(result["evaluation"])
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from agent_server.sec_extraction.config import ExtractionConfig, get_config
from agent_server.sec_extraction.schemas import ExtractionState
from agent_server.sec_extraction.tools.enrichment import enrich_record
from agent_server.sec_extraction.tools.evaluate import evaluate_record
from agent_server.sec_extraction.tools.llm_extract import llm_extract_record
from agent_server.sec_extraction.tools.retrieval import retrieve_context
from agent_server.sec_extraction.tools.scraper import scrape_attributes
from agent_server.sec_extraction.tools.text_extraction import extract_text
from agent_server.sec_extraction.tools.web_search import web_search

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node functions — each returns a partial state update dict
# ---------------------------------------------------------------------------


def text_extract_node(state: dict, config: RunnableConfig | None = None) -> dict:
    raw = state.get("raw_text", "")
    clean = extract_text(raw)
    logger.info("text_extract: %d chars → %d chars", len(raw), len(clean))
    return {"clean_text": clean}


def scraper_node(state: dict, config: RunnableConfig | None = None) -> dict:
    clean = state.get("clean_text", "")
    result = scrape_attributes(clean)
    logger.info("scraper: found %d fields via regex", len(result))
    return {"scraper_result": result}


def retrieval_node(state: dict, config: RunnableConfig | None = None) -> dict:
    ext_config = _get_ext_config(config)
    clean = state.get("clean_text", "")
    name = state.get("scraper_result", {}).get("business_name", "")
    query = f"{name}  Place Type:The type of place based on its role in the organization.,In Business:The operational status of the business.,Verification Date:The date the business was most recently verified.,Created At:The date and time when the place record was created.,Estimated Opened For Business:Estimates when the place opened for business.,Name:The place common, recognized name, or doing business as name.,Street:The location address of the place.,City:The city name for the Place.,State:The state or province of the place.,Postal Code:The postal code for the location address.,Primary Sic Code:The primary line of business represented by a Standard Industrial Classification code.,Primary Naics Code:The primary line of business represented by a North American Industry Classification System code.,Phone:The telephone number for the place.,Website:The primary homepage URL of the business.,Location Employee Count:The number of employees who work at this place.,Estimated Location Employee Count:Estimates the number of employees who work at this location.,Estimated Location Sales Volume:An estimation of the place sales revenue.,Contacts Count:The number of contacts on the business record.,Company ID:The unique 12-digit ID of the financial company.,Company Name:Primary name of the Company.,Company Legal Name:Legal name of the Company.,Company Legal Structure:Legal structure classification of the Company.,Company Type:Classification system for the Company.,Company Structure:The type of entity structure for the Company.,Company Ein:The employer identification number (EIN) of the Company.,Company Active Indicator:Indicates whether the company is active.,Company Legal Active Indicator:Indicates that the Company is legally active.,Company Operational Active Indicator:Indicates that the Company is operationally inactive.,Company Legal Entity Indicator:Indicates whether a Company has one or more associated legal entities.,Company Non Profit Indicator:Indicates whether the Company is non-profit.,Company Sp500 Indicator:Indicates whether the Company is a member of the S&P 500.,Company Delaware Stock Filings Indicator:Indicates whether the Company submits stock-related filings in Delaware.,Company Address:The Company headquarters address.,Company Suite:The Company headquarters unit or suite number.,Company City:City of the Company headquarters address.,Company State:State abbreviation of the Company headquarters address.,Company Fips:The unique identifier for counties based on the  FIPS code.,Company Postal Code:9-digit zip code (ZIP + 4) of the Company headquarters address.,Company Cbsa Code:The core based statistical area where the Company is located.,Company Sic Code:Primary SIC industry code.,Company Sic Name:Primary SIC industry name.,Company Naics Code:Primary NAICS industry code.,Company Naics Name:Primary NAICS industry name.,Company Phone:Primary company phone number.,Company Location Count:The total number of locations associated with the company.,Company Employment Count:Most recent U.S. full-time employee count.,Company Year Founded:The year the company was founded.,Company Date Dissoluted:Company dissolution date.,Company Confidence Score:Confidence score in company record accuracy.,Revenue:Total income from sales of goods or services.,Ebitda:Earnings before interest, taxes, depreciation, and amortization.,Cost Of Revenue:Direct costs of producing and delivering goods or services.,Net Income:Profit after all expenses, taxes, and interest.,Gross Profit:Revenue minus cost of revenue.,Total Assets:Sum of all current and non-current assets owned by the company.,Payroll:Total compensation owed to employees.,Operating Expenses:Costs of running daily business operations, excluding cost of revenue.,Operating Income:Profit from core operations after operating expenses.,Tax And Interest:Expenses related to income taxes and interest on debt.,Current Assets:Assets expected to be converted to cash, sold, or used within one year.,Cash:Liquid assets available for immediate use.,Credit Score:Numerical rating of a businesss creditworthiness.,Report Date:Date when financial or operational data is reported.,Other Current Assets:Miscellaneous short-term assets not classified elsewhere.,Non Current Assets:Assets not expected to be converted to cash within one year.,Loans To Shareholders:Funds advanced by the company to its shareholders.,Mortgage And Real Estate Loans:Loans secured by real estate, including mortgages and property financing.,Other Investments:Investments not classified as cash equivalents or core holdings.,Buildings And Other Depreciable Assets:Tangible fixed assets subject to depreciation.,Less Accumulated Depreciation:Total depreciation recorded against depreciable fixed assets.,Depletable Assets:Natural resources consumed over time.,Less Accumulated Depletion:Total depletion recorded against depletable natural resource assets.,Land Value:Value of owned land reported on the balance sheet.,Intangible Assets Amortizable:Intangible assets with finite lives, subject to amortization.,Less Accumulated Amortization:Value of intangible assets net of accumulated amortization over their useful life.,Other Non Current Assets:Miscellaneous long-term assets not classified elsewhere.,Total Liabilities And Equity:Sum of all liabilities and equity.,Current Liabilities:Obligations due within one year.,Accounts Payable:Amounts owed to suppliers and vendors for goods or services received on credit.,Short Term Debt:Borrowings and financial obligations due within one year.,Other Current Liabilities:Miscellaneous short-term obligations not classified elsewhere.,Non Current Liabilities:Financial obligations due beyond one year.,Loans From Shareholders:Funds borrowed by the company from its shareholders.,Long Term Debt:Borrowings and financial obligations due beyond one year.,Other Non Current Liabilities:Miscellaneous long-term obligations not classified elsewhere.,Shareholders Equity:Residual interest in assets after liabilities are deducted.,Total Debt:Sum of short-term and long-term borrowings.,Gross Profit Margin:Ratio of gross profit to revenue.,Ebitda Margin:Ratio of EBITDA to revenue.,Asset Turnover:Ratio of revenue to total assets.,Net Profit Margin:Ratio of net income to revenue.,Return On Assets:Ratio of net income to total assets.,Return On Sales:Ratio of net income to revenue.,Revenue Per Employee:Revenue generated per employee,Ebitda Per Employee:Earnings before interest, taxes, depreciation, and amortization.,Cost Of Revenue Per Employee:Average cost of revenue attributed to each employee.,Net Income Per Employee:Net income allocated per employee.,Gross Profit Per Employee:Gross profit allocated per employee.,Total Assets Per Employee:Value of total assets relative to employee count.,Payroll Per Employee:Average payroll expense per employee.,Operating Expenses Per Employee:Average operating costs per employee.,Operating Income Per Employee:Operating income divided by employee count.,Tax And Interest Per Employee:Tax and interest obligations allocated per employee.,Current Ratio:Ratio of current assets to current liabilities.,Debt To Equity Ratio:Ratio of total liabilities to equity.,Quick Ratio:Ratio of liquid current assets to current liabilities.,Trade Notes And Accounts Receivable:Amounts owed by customers through credit sales and trade notes.,Less Allowance For Bad Debts:Estimated uncollectible portion of accounts receivable.,Inventories:Goods and materials held for sale or production.,Us Government Obligations:Securities issued or guaranteed by the U.S. government.,Tax Exempt Securities:Investments generating income exempt from federal income taxes.,Revenue Growth Yoy:Year-over-year percentage change in total revenue.,Employment Growth Yoy:Year-over-year percentage change in total employees.,Revenue Growth Quarterly Yoy:Year-over-year percentage change in quarterly revenue.,Employment Growth Quarterly Yoy:Year-over-year percentage change in quarterly headcount.,Employment Growth Monthly Yoy:Year-over-year percentage change in monthly headcount.,Revenue Growth Qoq:Quarter-over-quarter percentage change in revenue.,Employment Growth Qoq:Quarter-over-quarter percentage change in total employees.,Employment Growth Mom:Month-over-month percentage change in total employees.,Work At Home:The place is a work at home business.,Duplicate Of:The IGID of the valid record that the place is a duplicate of.,Suppressed:The place is suppressed from customer feeds.,Suppressed Fields:The list of fields that are suppressed from display.,Suppressed Fields Count:The total number of Suppressed Fields associated with the place.,Updated At:The date and time when the place was last updated.,In Business Determined At:The date when the in business status was made.,In Business Research:The research performed to determine the in business status.,Opened For Business On:The date when the place opened for business.,Out Of Business On:The date the place went out of business.,Local Listings Claimed At:The date and time when the record was last updated by a Local Listings submission.,Local Listings Premium Claimed At:The date and time the record was last updated by a Local Listings Premium submission.,Alternative Name:An additional business name or DBA for the place.,Historical Names:Previously used names for the business.,Historical Names Count:The total number of Historical Names associated with the place.,Legal Names:The legal names of the Place.,Legal Names Count:Total number of legal names at a place,Suite:The unit or apartment number.,Country Code:The country code of the location.,Territory:Indicates whether place is located in a territory.,Cross Street Address:The intersection or cross street address of the location.,Address Type:The type of address for the location.,Address Changed On:The date when the business changed location.,Mailing Address:The PO Box or RR Box of the place.,Mailing Address City:The town or municipality where the place receives mail.,Mailing Address State:The state or province of the mailing address.,Mailing Address Postal Code:The postal code for the mailing address.,Mailing Address Type Code:The type of mailing address for the business.,Latitude:The latitude of the place.,Longitude:The longitude of the place.,Coordinate Match Level:The precision level of the latitude and longitude.,Geocoordinate:Latitude and longitude of the place.,Manual Geocoordinate:Pinpointed latitude and longitude for the physical building.,Landmark Address:The name of the complex, building, or mall where the place resides.,Location Linkage: Landmark ID:A unique ID assigned to a group of places at the same location.,Location Linkage: Parent:The primary business at a location with multiple places.,Location Linkage: Parent Relationship:The relationship to the primary business at a location.,Location Linkage: Place Count:The number of places at a location.,Location Linkage: Professional Count:The number of professionals at a location.,Location Linkage: Tenant Count:The number of verified places at a location.,Primary Sic Code Id Four Digit:The primary line of business represented by a 4-digit Standard Industrial Classification code.,Primary Sic Code Id Two Digit:The primary line of business represented by a 2-digit Standard Industrial Classification code.,SIC Codes:The lines of business represented by Standard Industrial Classification codes.,SIC Codes Count:The total number of SIC Codes associated with the place.,SIC Code Ids Four Digit:The lines of business represented by a 4-digit Standard Industrial Classification codes.,SIC Code Ids Four Digit Count:The total number of 4-digit SIC Codes associated with the place.,SIC Code Ids Two Digit:The lines of business represented by a 2-digit Standard Industrial Classification codes.,SIC Code Ids Two Digit Count:The total number of 6-digit SIC Codes associated with the place.,Primary Naics Code Id Four Digit:The primary line of business represented by a 4-digit North American Industry Classification System code.,Primary Naics Code Id Six Digit:The primary line of business represented by a 6-digit North American Industry Classification System code.,NAICS Codes:The lines of business represented by North American Industry Classification System codes.,NAICS Codes Count:The total number of NAICS Codes associated with the place.,Naics Code Ids Four Digit:The lines of business represented by a 4-digit North American Industry Classification System codes.,Naics Code Ids Four Digit Count:The total number of 4-digit NAICS Codes associated with the place.,Naics Code Ids Six Digit:The lines of business represented by a 6-digit North American Industry Classification System codes.,Naics Code Ids Six Digit Count:The total number of 6-digit NAICS Codes associated with the place.,Business Types:The specific types of business within a particular industry.,Business Types Count:The total number of Business Types associated with the place.,Toll Free Number:The toll-free phone number for the place.,Fax Number:The fax number for the place.,Additional Phone:An additional phone number for the place.,Corporate Email Address:The general contact email for the place.,Facebook URL:The link to the place on Facebook.,Twitter Url:The link to the place on Twitter.,Tiktok Url:The link to the place on Tiktok.,LinkedIn URL:The link to the place on LinkedIn.,Yelp Url:The link to the place on Yelp.,Pinterest Url:The link to the place on Pinterest.,Youtube Url:The link to the place YouTube channel or company video.,Tumblr Url:The link to the place on Tumblr.,Foursquare Url:The link to the place on Foursquare.,Instagram Url:The link to the place on Instagram.,Logo Url:The link to the place logo image.,Booking Url:The link to the place website to make a reservation or schedule an appointment.,Website Keywords:Keywords from company website HTML.,Website Keywords Count:The total number of Website Keywords associated with the place.,Primary Contact: ID:The unique ID for the contact.,Primary Contact: Created At:The date and time the contact detail was created.,Primary Contact: Email:The email address of the contact.,Primary Contact: Email Deliverable:The deliverability status of the email address.,Primary Contact: Email Marketable:The marketable status of the email address.,Primary Contact: Suppressed Fields:The fields on the individual that are suppressed.,Primary Contact: Suppressed Fields Count:The total number of Suppressed Fields associated with the contact.,Primary Contact: Standardized Job Titles:Standardized label of the individual job.,Primary Contact: Standardized Job Titles Count:The total number of Standardized Job Titles associated with the contact.,Primary Contact: First Name:The first name of the individual.,Primary Contact: Gender:The gender of the individual.,Primary Contact: Job Function:Describes the type of work an employee performs in their organization.,Primary Contact: Job Titles:Descriptive label of the individual job,Primary Contact: Job Titles Count:The total number of Job Titles associated with the contact.,Primary Contact: Last Name:The last name of the individual.,Primary Contact: Management Level:Describes the contact position relative to others in their organization.,Primary Contact: Mapped Contact:An identifier to map the individual to entries delivered by other Data Axle products.,Primary Contact: Professional Title:The professional degree or title of the individual.,Primary Contact: Primary:Information on whether the individual is the primary contact for the location.,Primary Contact: Title Codes:A list of the individual job titles.,Primary Contact: Title Codes Count:The total number of Title Codes associated with the individual.,Contacts: ID:The unique ID for the contact.,Contacts: Created At:The date and time the contact detail was created.,Contacts: Email:The email address of the contact.,Contacts: Email Deliverable:The deliverability status of the email address.,Contacts: Email Marketable:The marketable status of the email address.,Contacts: Suppressed Fields:The fields on the individual that are suppressed.,Contacts: Suppressed Fields Count:The total number of Suppressed Fields associated with the contact.,Contacts: Standardized Job Titles:Standardized label of the individual job.,Contacts: Standardized Job Titles Count:The total number of Standardized Job Titles associated with the contact.,Contacts: First Name:The first name of the individual.,Contacts: Gender:The gender of the individual.,Contacts: Job Function:Describes the type of work an employee performs in their organization.,Contacts: Job Titles:Descriptive label of the individual job,Contacts: Job Titles Count:The total number of Job Titles associated with the contact.,Contacts: Last Name:The last name of the individual.,Contacts: Management Level:Describes the contact position relative to others in their organization.,Contacts: Mapped Contact:An identifier to map the individual to entries delivered by other Data Axle products.,Contacts: Professional Title:The professional degree or title of the individual.,Contacts: Primary:Information on whether the individual is the primary contact for the location.,Contacts: Title Codes:A list of the individual job titles.,Contacts: Title Codes Count:The total number of Title Codes associated with the individual.,Ownership Changed On:The date when ownership of the place changed.,Headquarters:The direct headquarters of this place in its corporate family.,Ancestor Headquarters:The full list of all ancestors in the corporate family.,Ancestor Headquarters Count:The total number of Ancestor Headquarters associated with the place.,Ultimate Headquarters:The top-level headquarters of a corporate family.,Chain:The corporate chain for branch locations.,Corporate Franchising:Determine if the corporation includes franchised branches.,Corporate Employee Count:The actual, reported total number of employees at all locations in a corporate family.,Estimated Corporate Employee Count:Estimates the total number of employees at all locations in a corporate family.,Corporate Sales Revenue:The actual, reported sales revenue for the corporate family.,Estimated Corporate Sales Revenue:Estimates total sales revenue for the corporate family.,Branch Count:The number of branches reporting to a headquarter record.,Foreign Parent Flag:The place is owned by a corporation outside the US or Canada.,Fortune Ranking:The place ranking on the Fortune Magazine Top 1000 list.,Fiscal Year End Month:The month the place fiscal year ends.,Stock Exchange:The Stock Exchange where the corporation conducts trading activity.,Stock Ticker Symbol:The abbreviation used to identify the company on a stock market.,CIK:The Central Index Key (CIK) assigned to the corporation for filing with the SEC.,Affiliations:A list of organizations the place is affiliated with.,Affiliation Ids Count:The total number of Affiliation IDs associated with the place.,Brands:A list of brands sold by the business.,Brands Count:The total number of Brands associated with the place.,Company Description:The description of the business.,Dress Code:Details about the place dress code.,Equipment Rentals:Information on whether equipment rentals are available.,Has Ecommerce:The business is exchanging goods and services over the internet,Languages Spoken:The languages spoken at the place.,Languages Spoken Count:The total number of languages spoken at the place.,Price Range:The price range for products and services sold at the place.,Public Access:Indicates whether the place is publicly accessible.,Services Offered:The types of services offered at the place.,Store Number:ID assigned to identify a location within a corporation.,Unique Entity ID:The unique identifier for a place doing business with the U.S. government.,Federal Government Contractor:Indicates the Place is a contractor for the Federal government.,Owner Female:Information on whether the business is owned by a female.,Owner Minority:Information on whether the business is owned by a minority.,Owner Veteran:Information on whether the business is owned by a veteran.,Operating Hours Description:A description of the Operating Hours at the Place.,Operating Hours: Id:The unique ID for the Operating Hours detail.,Operating Hours: Created At:The date and time the current operating hours detail was created.,Operating Hours: Start Time:The time the place opens.,Operating Hours: End Time:The time the place closes.,Operating Hours: Days:The days that have operating hours.,Operating Hours: Days Count:The total number of days associated with the Operating Hours.,Operating Hours Count:The number of Operating Hours at the place.,Payment Types:The types of payments accepted at the location.,Payment Types Count:The total number of Payment Types accepted by the place.,Insurances Accepted:The types of insurances accepted by the business.,Insurances Accepted Count:The total number of insurances accepted by the place.,Medicare Accepted:Information on whether the place accepts Medicare.,Medicaid Accepted:Information on whether the place accepts Medicaid.,Images: Id:The unique ID for the Image detail.,Images: Created At:The date and time the current images detail was created.,Images: Asset Url:The full link to view and retrieve the image.,Images: Asset Hash:An ID used to view and retrieve the image.,Images: Primary:Information on whether the Image is the primary image for the place.,Images Count:The number of Images on the place record.,Service Area: Created At:The date and time the current service area detail was created.,Service Area: Postal Codes:The list of postal codes making up the service area of the place.,Service Area: Postal Codes Count:The total number of Postal Codes associated with the service area.,Service Area: Cities:The list of cities making up the service area of the place.,Service Area: Cities Count:The total number of Cities associated with the service area.,CBSA:The core based statistical area where the place is located.,CBSA Level:Information on whether the CBSA is a micropolitan or metropolitan area.,CSA:The combined statistical area code where the place is located.,Census Block Group:The census block group of the location as defined by the US Census Bureau.,Census Tract:The census tract of the location as defined by the US Census Bureau.,Dissemination Area:The dissemination area of the location as defined by Statistics Canada,Carrier Route:The carrier route for the location address as assigned by the USPS.,Mailing Address Carrier Route:The carrier route for the mailing address as assigned by the USPS.,FIPS Code:The unique identifier for counties based on the Federal Information Processing Standards (FIPS).,County Code:The 3-digit county code based on the place location ZIP code.,SGC Division:The unique identifier for counties based on the Standard Geographical Classification (SGC).,Delivery Point Barcode:The bar code for the postal route, assigned to the physical address.,Mailing Address Delivery Point Barcode:The bar code for the postal route, assigned to the mailing address.,Mailing Score Code:The deliverability score of the location address.,Mailing Address Score Code:The deliverability score of the mailing address.,ZIP Code:The 5-digit ZIP Code for the place.,Zip Four:The zip+4 code for the place.,Mailing Address Zip:The 5-digit ZIP code for the mailing address.,Mailing Address Zip Four:The 4-digit ZIP extension for the mailing address.,CMRA:Indicates whether the location is a Commercial Mail Receiving Agency.,Mailing Address CMRA:Information on whether the mailing address listed for the place is a Commercial Mail Receiving Agency.,Neighborhood:The neighborhood of the location.,Parsed Primary Address:Address without unit type and number,Parsed House Number:House number of the parsed address,Parsed Pre Direction:Pre directional of the parsed address,Parsed Street Name:Street name of the parsed address,Parsed Street Suffix:Street suffix of the parsed address,Parsed Post Direction:Post directional of the parsed address,Parsed Unit Type Number:Unit type and number of the parsed address,Parsed Unit Number:Unit number of the parsed address,Parking: Created At:The date and time the current parking detail was created.,Parking: Bike Rack:A bike rack is available.,Parking: Bike Lockers:Bike lockers are available.,Parking: Electric Charging Port Count:The number of electric charging ports at the location.,Parking: Free:Information on whether parking is free.,Parking: Number Of Spaces:The number of parking spaces available.,Parking: Onsite:The place has on-site parking.,Parking: Overnight Parking:Overnight parking is available.,Parking: Permit Required:A parking permit is required.,Parking: Public:Information on whether public parking is available.,Parking: Transit Lines:Transit lines that run to and from the Place.,Parking: Valet:The place offers valet parking.,Credit: Created At:The date and time the current credit detail was created.,Credit: Grade:An estimation of a Place creditworthiness in letter grade format.,Credit: Limit:The recommended credit limit based on an estimation of the Place credit worthiness.,Credit: Trend:Indicates whether a business credit score is rising, steady, or decreasing.,Bankruptcies: Created At:The date and time the current bankruptcy detail was created.,Bankruptcies: Case Number:The case number associated with the bankruptcy.,Bankruptcies: Dismissal:The debtor is trying to dismiss the case.,Bankruptcies: Filing Date:The date the public record was first filed.,Bankruptcies: Release Date:The date the bankruptcy public record was released.,Bankruptcies: Filing Type:The type of bankruptcy filing.,Bankruptcies Count:Total number of bankruptcies associated with the place.,Benefit Plans: Id:The unique ID for the benefit plan.,Benefit Plans: Created At:The date and time the current benefit plan detail was created.,Benefit Plans: Active Beginning Participants:Number of active participants at the beginning of the plan year.,Benefit Plans: Active Ending Participants:Number of active participants at the end of the plan year.,Benefit Plans: Benefit Arrangements:Benefit arrangements for the plan.,Benefit Plans: Benefit Arrangements Count:The total number of benefit arrangements on the plan.,Benefit Plans: Broker City:City of the agent or broker.,Benefit Plans: Broker Name:Name of the agent or broker to whom commissions/fees are paid.,Benefit Plans: Broker Postal Code:Postal code of the agent or broker.,Benefit Plans: Broker State:State of the agent or broker.,Benefit Plans: Broker Street:Address of the agent or broker.,Benefit Plans: Carrier:Name of the insurance carrier.,Benefit Plans: Contract Number:Contract number of the plan.,Benefit Plans: Effective Date:Effective date of the plan.,Benefit Plans: Funding Arrangements:Funding arrangements for the plan.,Benefit Plans: Funding Arrangements Count:The total number of funding arrangements on the plan.,Benefit Plans: Name:Name of the benefit plan.,Benefit Plans: Short Form:Indicates the short form for small businesses.,Benefit Plans: Term:Beginning and end dates of the plan year.,Benefit Plans: Total Beginning Participants:Total number of participants at the beginning of the plan year.,Benefit Plans: Welfare Benefit Types:Types of welfare benefits of the plan.,Benefit Plans: Welfare Benefit Types Count:The total number of types of welfare benefits on the plan.,Benefit Plans Count:Total number of benefit plans associated with the place.,Corporation File Type:The type of business registration filing.,Incorporation Date:Date when place registered as a business.,Vehicle and Equipment Makes:The make of car that a car dealer sells.,Vehicle and Equipment Makes Count:The total number of Vehicle and Equipment Makes associated with the place.,Religious Denominations:The religious denomination of the church or other house of worship.,Religious Denominations Count:The total number of Religious Denominations associated with the place.,Restaurant: Created At:The date and time the current restaurant detail was created.,Restaurant: Cuisines:The type of cuisine served at the place.,Restaurant: Cuisines Count:The total number of cuisines associated with the restaurant.,Restaurant: Delivery:Indicates the restaurant offers delivery service.,Restaurant: Dining Options:The types of dining options available at the restaurant.,Restaurant: Dining Options Count:The number of dining options available at the restaurant.,Restaurant: Limited Service:Indicates if the Place is a full or limited service restaurant.,Restaurant: Menu Url:The link to view the restaurants menu.,Restaurant: Online Order Url:The link to place an online order.,Restaurant: Reservations:The restaurant has reservations.,Restaurant: Service Options:The types of service options available at the restaurant.,Restaurant: Service Options Count:The number of service options available at the restaurant.,Restaurant: Takeout:The restaurant has takeout.,Happy Hours: Id:The unique ID for the Happy Hour.,Happy Hours: Created At:The date and time the current happy hour detail was created.,Happy Hours: Special Food:Information on whether the Happy Hour has food specials.,Happy Hours: Special Drink:Information on whether the Happy Hour has drink specials.,Happy Hours: Special Activity:Information on whether the Happy Hour has special activities.,Happy Hours: Special Other:Information on whether the Happy Hour has additional specials.,Happy Hours: Start Time:The Happy Hour start time.,Happy Hours: End Time:The Happy Hour ending time.,Happy Hours: Days:The days when Happy Hour is offered.,Happy Hours: Days Count:The total number of days associated with the Happy Hours.,Happy Hours: Description:The Happy Hour description.,Happy Hours Count:The number of Happy Hour records on the place.,Hotel: Created At:The date and time the current hotel detail was created.,Hotel: Cable Tv:The hotel has Cable TV.,Hotel: Continental Breakfast:The hotel offers continental breakfast.,Hotel: Elevator:The hotel has an elevator.,Hotel: Exercise Facility:The hotel has an exercise facility.,Hotel: Guest Laundry:The hotel has guest laundry services.,Hotel: Hot Tub:The hotel has a hot tub.,Hotel: Indoor Pool:The hotel has an indoor pool.,Hotel: Kitchens:The hotel has guest kitchens.,Hotel: Outdoor Pool:The hotel has an outdoor pool.,Hotel: Pet Friendly:The hotel allows pets.,Hotel: Room Service:The hotel offers room service.,Hospital Has Emergency Room:The hospital has an emergency room.,Professional: Id:The unique ID for the professional detail.,Professional: Created At:The date and time the current professional detail was created.,Professional: Graduation Year:The year the professional graduated with an advanced degree.,Professional: License Number:The professional license number.,Professional: Primary Specialty:The professional primary medical specialty.,Professional: Specialty Ids:The professional list of medical specialties.,Professional: Specialty Ids Count:Total number of specialties associated with the professional.,Professional: State Of License:The state where the professional license was issued.,Professional: Year Of Birth:The professional year of birth.,Provider: Id:The unique ID for the provider detail.,Provider: Created At:The date and time the current provider detail was created.,Provider: Accepting New Patients:The provider is accepting new patients.,Provider: Affiliated Hospital:The hospital the provider is associated with.,Provider: Board Certified:Information on whether the provider is board certified.,Provider: Dea Number:The provider DEA number.,Provider: Medical School:The medical school the provider attended.,Provider: National Provider Number:The provider National Provider Identifier.,Provider: Residency Graduation Year:The year the provider graduated from their residency program.,Provider: Residency Hospital:The hospital where the provider completed their residency.,Estimated Fleet Size:Estimates the number of vehicles owned or used by the location.,Greenscore:The propensity for the business to be a green adopter.,Growing Business Indicator:Information on whether the business is growing or shrinking a significant percentage.,White Collar Percentage:An estimation of white collar workers at the place.,Estimated Owns Location:Estimates whether a business owns or rents the business property,Estimated Square Footage:Estimates the square footage of the place.,Expenses: Created At:The date and time the current expense(s) detail was created.,Expenses: Accounting:Estimates the total cost of accounting, auditing, and bookkeeping services purchased from other companies.,Expenses: Advertising:An estimate of the cost of purchased advertising and promotional services.,Expenses: Charities:An estimate of the cost of charitable contributions.,Expenses: Contract Labor:An estimate of the cost of payments to other companies for the contractual use of their employees.,Expenses: Corporate:Information on whether the expense models are based on the corporation or location.,Expenses: Insurance:An estimate of the cost of payments made on insurance policies.,Expenses: Legal:An estimate of the cost of legal services purchased from other firms.,Expenses: Licenses:An estimate of the cost of license fees.,Expenses: Maintenance:An estimate of the cost of maintenance services.,Expenses: Office Equipment Supplies:An estimate of the cost of supplies, materials and parts purchased for a Place own use.,Expenses: Packaging Shipping:An estimate of the cost of packaging and shipping supplies.,Expenses: Payroll:An estimate of the cost of gross earnings of all employees for the calendar year.,Expenses: Printing:An estimate of the cost of purchased or contracted printing services.,Expenses: Professional Services:An estimate of the cost of management, consulting, administrative, and other professional services purchased from other companies.,Expenses: Rent Lease:An estimate of the cost of payments made to other companies for the rental or leasing.,Expenses: Technology:An estimate of the cost of technology supplies, materials, and development.,Expenses: Telecommunications:An estimate of the cost for communication services purchased from other companies.,Expenses: Transportation:An estimate of the cost of purchased or contracted transportation services.,Expenses: Utilities:An estimate of the cost of electricity and fuels for heating, power, or generation of electricity.,Population Density:The actual population density of the city.,Population Code For Zip:The population size of the city by ZIP code.,Wealthy Area Flag:An estimation on whether the place is located in a wealthy area." if name else clean[:200]
    chunks_df = retrieve_context(query, ext_config)
    logger.info("retrieval: got %d chunks from vector search", chunks_df.count())
    return chunks_df


def llm_extract_node(state: dict, config: RunnableConfig | None = None) -> dict:
    ext_config = _get_ext_config(config)
    record = llm_extract_record(
        text=state.get("clean_text", ""),
        scraper_result=state.get("scraper_result", {}),
        context_chunks=state.get("retrieval_context", []),
        config=ext_config,
    )
    logger.info("llm_extract: produced record with %d non-null fields",
                sum(1 for v in record.values() if v is not None and v != []))
    return {"record": record}


def enrichment_node(state: dict, config: RunnableConfig | None = None) -> dict:
    record = dict(state.get("record", {}))
    enriched = enrich_record(record, state.get("clean_text", ""))
    logger.info("enrichment: is_manufacturer=%s, is_open=%s",
                enriched.get("is_manufacturer"), enriched.get("is_open"))
    return {"record": enriched}


def evaluate_node(state: dict, config: RunnableConfig | None = None) -> dict:
    ext_config = _get_ext_config(config)
    evaluation = evaluate_record(
        record_dict=state.get("record", {}),
        source_text=state.get("clean_text", ""),
        config=ext_config,
    )
    logger.info("evaluate: fill_rate=%.2f, valid=%s",
                evaluation.get("fill_rate", 0), evaluation.get("valid"))
    return {
        "evaluation": evaluation,
        "fill_rate": evaluation.get("fill_rate", 0),
    }


def web_fallback_node(state: dict, config: RunnableConfig | None = None) -> dict:
    ext_config = _get_ext_config(config)
    name = state.get("record", {}).get("business_name", "")
    missing = state.get("evaluation", {}).get("missing_fields", [])
    query = f"{name} {' '.join(missing[:3])} SEC filing"
    snippets = web_search(query, ext_config)
    logger.info("web_fallback: got %d snippets for '%s'", len(snippets), query[:60])

    all_context = state.get("retrieval_context", []) + snippets
    record = llm_extract_record(
        text=state.get("clean_text", ""),
        scraper_result=state.get("scraper_result", {}),
        context_chunks=all_context,
        config=ext_config,
    )
    enriched = enrich_record(record, state.get("clean_text", ""))
    return {"record": enriched, "web_context": snippets, "iteration": 1}


def re_evaluate_node(state: dict, config: RunnableConfig | None = None) -> dict:
    ext_config = _get_ext_config(config)
    evaluation = evaluate_record(
        record_dict=state.get("record", {}),
        source_text=state.get("clean_text", ""),
        config=ext_config,
    )
    logger.info("re_evaluate: fill_rate=%.2f (after web fallback)", evaluation.get("fill_rate", 0))
    return {
        "evaluation": evaluation,
        "fill_rate": evaluation.get("fill_rate", 0),
    }


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------


def should_fallback(state: dict) -> str:
    """Route to web_fallback if fill rate is below threshold and we haven't tried yet."""
    fill_rate = state.get("fill_rate", 0)
    iteration = state.get("iteration", 0)
    threshold = 0.5  # overridden at runtime via config if needed

    if fill_rate < threshold and iteration < 1:
        return "web_fallback"
    return "end"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def _get_ext_config(config: RunnableConfig | None) -> ExtractionConfig:
    """Extract ExtractionConfig from LangGraph configurable or use default."""
    if config and "configurable" in config:
        ext = config["configurable"].get("ext_config")
        if ext:
            return ext
    return get_config()


def build_extraction_workflow(ext_config: ExtractionConfig | None = None) -> Any:
    """Compile the LangGraph extraction pipeline.

    Returns a compiled StateGraph that accepts ExtractionState.
    """
    graph = StateGraph(dict)

    graph.add_node("text_extract", text_extract_node)
    graph.add_node("scraper", scraper_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("llm_extract", llm_extract_node)
    graph.add_node("enrichment", enrichment_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("web_fallback", web_fallback_node)
    graph.add_node("re_evaluate", re_evaluate_node)

    graph.set_entry_point("text_extract")
    graph.add_edge("text_extract", "scraper")
    graph.add_edge("scraper", "retrieval")
    graph.add_edge("retrieval", "llm_extract")
    graph.add_edge("llm_extract", "enrichment")
    graph.add_edge("enrichment", "evaluate")

    graph.add_conditional_edges(
        "evaluate",
        should_fallback,
        {"web_fallback": "web_fallback", "end": END},
    )
    graph.add_edge("web_fallback", "re_evaluate")
    graph.add_edge("re_evaluate", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def extraction_workflow(
    document: str,
    config: ExtractionConfig | None = None,
) -> dict:
    """Run the full extraction pipeline on a single document.

    Args:
        document: Raw SEC filing text or HTML.
        config:   Pipeline configuration (uses defaults if None).

    Returns:
        Dict with keys: record, evaluation, fill_rate, and more.
    """
    cfg = config or get_config()
    workflow = build_extraction_workflow(cfg)

    initial_state = {
        "raw_text": document,
        "clean_text": "",
        "scraper_result": {},
        "retrieval_context": [],
        "record": {},
        "evaluation": {},
        "fill_rate": 0.0,
        "web_context": [],
        "iteration": 0,
    }

    result = workflow.invoke(
        initial_state,
        config={"configurable": {"ext_config": cfg}},
    )
    return result
