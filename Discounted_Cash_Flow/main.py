from finvizfinance.quote import *
from dotenv import load_dotenv
from datetime import datetime
from scipy import stats 
import yfinance as yf
import numpy as np
import pandas as pd
import requests
import time
import csv
import os
import re

# Source
# https://pages.stern.nyu.edu/~adamodar/New_Home_Page/valquestions/growth.htm#:~:text=The%20reinvestment%20rate%20for%20a,the%20course%20of%20the%20year.


class DiscountedCashFlows:
    def __init__(self, symbol, fmp_api_key, fred_api_key): 
        self.symbol = symbol
        self.fmp_api_key = fmp_api_key
        self.fred_api_key = fred_api_key
        self.yf_finance = yf.Ticker(self.symbol) 


    def request(version=None, endpoint=None, symbol=None, period=None, exceptions=None):
        try:
            if symbol is None:
                raise ValueError("Ticker symbol is required.")

            if exceptions is not None:
                url = exceptions
            else:
                if version is None or endpoint is None:
                    raise ValueError("Both version and endpoint are required if exceptions URL is not provided.")
                period_str = f'&period={period}' if period and (period == 'quarterly' or period == 'annual') else ''
                url = f"https://financialmodelingprep.com/api/{version}/{endpoint}/{symbol}?apikey={fmp_api_key}{period_str}"

            print(f"Request URL: {url}")

            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()
                return data
            else:
                print(f"Request failed with status code: {response.status_code}")
                return None

        except Exception as e:
            print(f"Error occurred while processing {symbol}: {e}")
    

    def global_data(self):

        """This function creates global variables that can be reused throughout the program"""

        # Balance Sheet 

        global balance_sheet

        balance_sheet = pd.DataFrame(self.request('v3', 'balance-sheet-statement', 'quarterly'))

        # Cash Flow Statement

        global cash_flow_statement  

        cash_flow_statement = pd.DataFrame(self.request('v3', 'cash-flow-statement', 'quarterly'))

        # Income Statement 

        global income_statement

        income_statement = pd.DataFrame(self.request('v3', 'income-statement', 'quarterly'))

        # Global DCF Data

        global profile 
        profile = pd.DataFrame(self.request('v3', 'profile', self.symbol))

        global perpetual_growth_rate
        perpetual_growth_rate =  yf.download("^IRX")["Adj Close"].iloc[-1]
        
        global current_year
        current_year = time.localtime().tm_year 

        # Grab Company Peers

        global peer_list

        peer_list = self.request('v4', 'stock_peers')
        peer_list_df = pd.DataFrame(peer_list)
        peer_list = peer_list_df.loc['peersList']

        global average_tax_rate
        
        tax_rates = income_statement['incomeTaxExpense'] / income_statement['incomeBeforeTax']
        
        average_tax_rate = np.mean(tax_rates)


    def load_country_abbreviations(csv_file):
        country_dict = {}
        with open(csv_file, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                country_dict[row[1].strip()] = row[0].strip()
        return country_dict


    def industry_beta(self):

        # Industry Beta
        market_caps = []
        industry_betas = []
        peer_list = []

        for stock in peer_list:

            # Grab levered beta
            profile_data = pd.DataFrame(self.request('v3', 'profile', stock))
            beta = profile_data.loc['beta']

            # Grab quarterly balance sheet
            balance_sheet = pd.DataFrame(self.request('v3', 'balance-sheet-statement', stock, 'quarterly'))

            # Grab quarterly income statment
            income_statement = pd.DataFrame(self.request('v3', 'income-statement', stock, 'quarterly'))

            # Tax, Debt & Equity info
            tax_rate = (income_statement.loc['incomeTaxExpense'].iloc[0]) / (income_statement.loc['incomeBeforeTax'].iloc[0])

            debt = balance_sheet.loc['totalDebt']
            equity = balance_sheet.loc['totalEquity']

            # Unlever the beta
            beta_unlevered = beta / (1 + (1-tax_rate) * (debt / equity ))
            mkt_cap = profile_data.loc['mktCap']
            industry_betas.append(beta_unlevered)
            market_caps.append(mkt_cap)

        # Calculate unlevered industry beta
        unlevered_industry_beta = sum(beta * cap for beta, cap in zip(industry_betas, market_caps))

        return unlevered_industry_beta
    

    def reinvestment_rate(self, ticker):
        # Reinvestment Rate Analysis

        balance_sheet = pd.DataFrame(self.request('v3', 'balance-sheet-statement', self.symbol, 'annual')).set_index('date')
        income_statement =  pd.DataFrame(self.request('v3', 'income-statement', self.symbol, 'annual')).set_index('date')
        cash_flow_statement = pd.DataFrame(self.request('v3', 'cash-flow-statement', self.symbol, 'annual')).set_index('date')

        if balance_sheet and income_statement and cash_flow_statement:
            reinvestment_rate_list = []

            for date in cash_flow_statement.index[:20]:
                depreciation_and_amortization = income_statement.loc[date, 'depreciationAndAmortization']
                net_capex = cash_flow_statement.loc[date, 'capitalExpenditure'] - depreciation_and_amortization
                change_in_working_capital = cash_flow_statement.loc[date, 'changeInWorkingCapital']
                ebitda = income_statement.loc[date, 'ebitda']
                tax_rate = (income_statement.loc[date, 'incomeTaxExpense']) / (income_statement.loc[date, 'incomeBeforeTax'])
                ebit = ebitda - depreciation_and_amortization
                nopat = ebit * (1 - tax_rate)

                reinvestment_rate = (net_capex + change_in_working_capital) / nopat
                reinvestment_rate_list.append((date, reinvestment_rate))
        else:
            print(f"ERROR WITH REINVESTMENT DATAFRAMES {ticker}")
            exit(1)
            
        return reinvestment_rate_list
    

    def final_reinvestment_rate(self):
        # Reinvestment Rate is calculated through industry rates & average historical rate
        
        reinvestment_rate_final = 0
        industry_reinvestment_rates = []

        # Historical Trimmed Mean Reinvestment Rate
        reinvestment_average = stats.trim_mean(self.reinvestment_rate(self.symbol), 0.5) # Trim 5% at both ends
        
        # Grab weighted average industry reinvestment rate + trimmed mean average historical reinvestment rate
        print(f"Average Reinvestment Rate for {self.symbol} = {reinvestment_average}")

        for stock in peer_list:

            rr = self.reinvestment_rate(stock)
            industry_reinvestment_rates.append(rr)

        industry_reinvestment_average = stats.trim_mean(industry_reinvestment_rates, 0.1) # Trim 10% at both ends
        print(f"Average Industry Reinvestment Rate for {self.symbol} = {industry_reinvestment_average}")
        reinvestment_rate_final = industry_reinvestment_average - reinvestment_average

        return reinvestment_rate_final


    def timeframe(self):	
        # Finding "appropiate" time frame of the DCF
        # Startup / Hyper Growth / Self-Funding / Operating Leverage / Capital Return / Decline
        # Discounted Cash Flows work best in Stage 4 & 5.

        ten_year_dcf = 0
        five_year_dcf = 0
        
        # Revenue Growth Variability 
        income_statement = self.request('v3', 'income-statement', self.symbol, 'quarterly')
        income_statement_df = pd.DataFrame(income_statement).set_index('date')
        income_statement_df['revenue'] = income_statement_df['revenue'].astype(float)

        revenue_growth_list = []
        for i in range(1, len(income_statement_df.index[:10])):
            current_revenue = income_statement_df.iloc[i]['revenue']
            previous_revenue = income_statement_df.iloc[i-1]['revenue']
            rev_growth = (current_revenue - previous_revenue) / previous_revenue
    
            revenue_growth_list.append(rev_growth)

        revenue_growth_series = pd.Series(revenue_growth_list)
        revenue_growth_variability = revenue_growth_series.std()

        print(f'Revenue Growth Variablity = {revenue_growth_variability}')

        # If the company operates in an industry with high volatility or rapid change, long-term projections may be highly uncertain and less reliable.
        if self.industry_beta() > 1.5:
            five_year_dcf += 1

        # Startups or early-stage companies with unpredictable growth trajectories often have less reliable forecasts beyond 5 years.

        if 'description' in profile.index:
            description = profile.loc['description']

            # Use regex to find the incorporation year in the description
            match = re.search(r'incorporated in (\d{4})', description)

            if match:
                incorporation_year = int(match.group(1))

                # Calculate the number of years since incorporation
                stock_lifetime = current_year - incorporation_year
            else:
                print("Incorporation year not found in the description.")
                if 'ipoDate' in profile.index:
                    ipo_date_string = profile.loc['ipoDate']
                    ipo_year = datetime.strptime(ipo_date_string, "%Y-%m-%d").year

                    # Calculate the stock lifetime in years
                    stock_lifetime = current_year - ipo_year
                else:
                    print("IPO Date is missing or invalid.")
        
            # Increment five_year_dcf if the stock lifetime is less than 5 years
            if stock_lifetime < 5:
                five_year_dcf += 1
        else:
            pass
        
        # Print the extracted year and stock lifetime
        print(f"Incorporation Year: {incorporation_year}")
        print(f"Stock Lifetime: {stock_lifetime} years")
        

        # If there's insufficient historical data to make credible long-term forecasts, a shorter horizon might be more appropriate.
        cash_flow_statement = pd.DataFrame(self.request('v3', 'cash-flow-statement', self.symbol, 'annual')).set_index('date')
        if range(cash_flow_statement.index) < 3:
            print(f'less than 3 years of historical data adding extra point for 5 year dcf')
            five_year_dcf +=1

        else:
            print(f'more than 3 years of historical data adding extra point for 10 year dcf')
            ten_year_dcf +=1


        data_table = pd.DataFrame(self.request('v3', 'income-statement', self.symbol, 'annual'))

        data_table.set_index('date', inplace=True)
        
        previous_revenue = None

        data_table = data_table.tail(3)

        rev_growth_list = []

        for index, row in data_table.index[:20].iterrows():
            revenue_current = row['revenue']

            if previous_revenue is not None:
                revenue_change = (revenue_current - previous_revenue) / previous_revenue
                rev_growth_list.append(index, revenue_change)

                print(f"Date: {index}, Revenue: {revenue_current}, Revenue Change: {revenue_change:.2%}")
            
            previous_revenue = revenue_current

        average_revenue_change = sum(rev_growth_list) / len(rev_growth_list)
        print(f"Average Revenue Change: {average_revenue_change:.2%}")

        # Increasing share count 

        url = f'https://financialmodelingprep.com/api/v4/historical/shares_float?symbol={self.symbol}&apikey={fmp_api_key}'
        
        increasing_share_count = pd.DataFrame(self.request(exceptions=url))

        # Ensure the 'date' column is in datetime format
        increasing_share_count['date'] = pd.to_datetime(increasing_share_count['date'])

        # Sort the DataFrame by date
        increasing_share_count = increasing_share_count.sort_values('date')

        # Extract share counts for the dates a year apart
        start_date = increasing_share_count['date'].min()
        end_date = start_date + pd.DateOffset(years=1)

        # Get the closest available data for start_date and end_date
        start_share_count = increasing_share_count.loc[increasing_share_count['date'] <= start_date, 'outstandingShares'].iloc[-1]
        end_share_count = increasing_share_count.loc[increasing_share_count['date'] <= end_date, 'outstandingShares'].iloc[-1]

        # Calculate the percentage change
        percentage_change = ((end_share_count - start_share_count) / start_share_count) * 100

        print(f"The share count changed by {percentage_change:.2f}% over the year.")

        # Company in decline (lower Revenue / Gross Profit / Operating Profit / Net Profit / Increasing Share Count)
        if average_revenue_change <= 0 and increasing_share_count > 0:
            five_year_dcf += 1

        # Companies with long-term capital investments and infrastructure projects, such as utilities or real estate, may require a longer projection period to accurately reflect their cash flows.
        sector = profile.loc["sector"]

        if sector == "Real Estate" or sector == "Utilities":
            print(f'Company in mature and stable sector. Sector of {self.symbol} == {sector}')
            ten_year_dcf +=1
        else:
            print(f'Company not in mature and stable sector. Sector of {self.symbol} == {sector}')
            five_year_dcf +=1
        
        # Industries with long product development and life cycles (e.g., pharmaceuticals, aerospace) might need a longer DCF period to capture the return on their investments.
        industry = profile.loc["industry"]
         
        if industry == "Aerospace & Defense" or industry == "Biotechnology" or industry == 'Drug Manufacturers - General':
            print(f'Company in industry with long product development Industry of {self.symbol} == {industry}')
            ten_year_dcf +=1
        else:
            print(f'Company not in long product development and life cycles industry of {self.symbol} == {industry}')
            five_year_dcf +=1

        # High-growth companies that are expected to scale significantly over a decade may require a longer projection period to capture the expected expansion in their cash flows.
        global timeframe
            
        if ten_year_dcf < five_year_dcf:
            print("5-year DCF is more appropriate due to high volatility.")
            timeframe = '5'
            return timeframe
        else:
            print("10-year DCF is more appropriate due to stable conditions.")
            timeframe = '10'
            return timeframe
    

    def wacc(self):
            
            balance_sheet = pd.DataFrame(self.request('v3', 'balance-sheet-statement', self.symbol, 'quarterly'))
            income_statement = pd.DataFrame(self.request('v3', 'income-statement', self.symbol, 'quarterly'))
            
            debt = balance_sheet.loc['totalDebt']
            equity = balance_sheet.loc['totalEquity'] 

            tax_rate = (income_statement.loc['incomeTaxExpense'].iloc[0]) / (income_statement.loc['incomeBeforeTax'].iloc[0])

            debt_to_equity_ratio = debt / equity

            cost_of_debt = income_statement.loc['interestExpense'] / debt 

            wacc_list = []

            # While I realize this should be the weighted market risk premium for the segemented revenue stream- 
            # it is not achievable in the data provided by Financial Modeling Prep

            country = profile.loc["country"]
            
            csv_file = 'countries.csv'
            
            country_mapping = self.load_country_abbreviations(csv_file)
            if country in country_mapping:
                country_name = country_mapping[country]
            else:
                print(f'Cannot find {country} in Country Mapping CSV File')
            
            equity_risk_premium = pd.DataFrame(self.request('v4', 'market_risk_premium', self.symbol)).loc[country_name]

            # 3 Month T-Bill for Risk Free Rate

            risk_free_rate = yf.download("^IRX")["Adj Close"].iloc[-1]

            relevered_beta = self.industry_beta() * (1 + debt_to_equity_ratio * (1 - tax_rate))

            predicted_rfr_list = []

            for i in range(timeframe):

                # The assumption being in this example which ever the country that the company resides in will be the equity risk premium

                cost_of_equity = risk_free_rate + relevered_beta * equity_risk_premium                
            
                credit_risk = cost_of_debt - risk_free_rate

                # Prediction of cost of debt for future risk free rates 



                # CME Watch Tool API Coming soon



                # Historically speaking, actual rates tend to overshoot both the forward curve
                # The Fed dot plot in a raising interest rate environment and rates tend to drop faster than the the Fed dot plot had predicted
                # Although the Fed projections and the forward curve are the best estimate we can make for future rates, neither tend to be historically accurate.



                # Fed Dot Plot FRED API

                url = f"https://fred.stlouisfed.org/series/FEDTARMD/?apikey={self.fred_api_key}"
                
                prediction_year = i + current_year



                print(f"Request URL: {url}") 

                response = requests.get(url)

                if response.status_code == 200:
                    data = response.json()
                    return data
                
                predicted_rfr = 

                predicted_rfr_list.append(predicted_rfr)
                print(f'WACC for {prediction_year} == {predicted_rfr}')

                if i == timeframe:
                        cost_of_debt
                else:
                    cost_of_debt = predicted_rfr_list[i] + credit_risk

                
                print(f'Cost of Debt in {i + current_year} for {self.symbol} is {cost_of_debt}')
                print(f'Cost of Equity in {i + current_year} for {self.symbol} is {cost_of_equity}')

                # Calculate WACC & Relever beta
                equity_weight = 1 / (1 + debt_to_equity_ratio)
                debt_weight = debt_to_equity_ratio / (1 + debt_to_equity_ratio)
                wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt * (1 - tax_rate))  

                wacc_list.append(wacc)
                print(f"Wacc for Year {i + current_year}")


            return wacc_list
    

    def discounted_cashflows(self):

            # Analyst Estimates
            version = 'v3'
            endpoint = 'analyst-estimates'

            estimate_data = self.request(version, endpoint)
            estimate_data = pd.DataFrame(estimate_data).set_index('date')
            
            indexed_year = current_year

            # Checking to see length of estimates
            for entry in estimate_data:

                if indexed_year in entry['date']:
                    indexed_year + 1
                else:
                    analyst_timeframe = indexed_year - current_year
                    print(f'Total Years of Analyst Estimation = {(analyst_timeframe)}')

            if analyst_timeframe < 5:
                print(f'Calculating / estimating next {analyst_timeframe - self.timeframe}')
                # Financials for Reinvestment Analysis
                # (Net Capex + Change in Net Working Capital) / (Net Operating Profit After Taxes)

                # ROE method of growth estimation


                # EBIT estimate
                # Reinvestment Rate * Return on Capital


            else:

                pass


            fcff_discounted = []


            for year in analyst_timeframe:
                ebit_average = estimate_data.loc['estimatedEbitAvg'][year]


                reinvestment_rate = self.final_reinvestment_rate()[0]



            for values, in ebit_average:
                fcff = ((values *(1-tax_rate[i])) * (1-reinvestment_rate) ) * (1 + wacc[values])


                fcff_discounted.append(fcff)


            global final_year_fcff
            final_year_fcff = fcff_discounted[-1]


            return fcff_discounted


    def terminal(self):
        final_year = len(timeframe)
        terminal_fcff = (self.discounted_cashflows[final_year] * (1 + perpetual_growth_rate) /((self.wacc[final_year])^final_year))
        return terminal_fcff


    def assumptions(self):
        ebit_margins =
        rev_growth = 
        tax_rate = 
        fcff = 

        print(f'Assumptions for Automated DCF of {self.symbol} are as follows:')
        print(f'Expected Ebit Margins == {ebit_margins}')
        print(f'Expected Rev Growth == {rev_growth}')
        print(f'Expected Tax Rate == {(tax_rate * 100):.2%}')
        print(f'Expected FCFF for the next {timeframe} years == {sum(self.discounted_cashflows)}')
        print(f'Terminal FCFF Growth = {perpetual_growth_rate:.2%}')


    def fair_value(self):
        try:

            self.global_data
            self.industry_beta
            self.reinvestment_rate
            self.timeframe
            self.wacc
            
            self.terminal
            self.assumptions

            terminal_fcff = (final_year_fcff * (1+ self.wacc[-1])) / (self.wacc[-1]) / ()
            
            # Grab from Yahoo Finance, Firm Value & Current Share Count
            non_current_assets = self.yf_finance.quarterly_balance_sheet.loc['Total Non Current Assets'].iloc[0]
            net_liabilities = self.yf_finance.quarterly_balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0]
            cash = self.yf_finance.quarterly_balance_sheet.loc['Cash Cash Equivalents And Short Term Investments'].iloc[0]
            shares_outstanding = self.yf_finance.quarterly_balance_sheet.loc['Ordinary Shares Number'].iloc[0]

            equity_value = sum(self.discounted_cashflows) + terminal_fcff + non_current_assets + cash - net_liabilities 
            fair_value = equity_value / shares_outstanding

            option_upside = 'N/A'

            ntm_price_target = pd.DataFrame(self.request('v4', 'price-target-summary')).iloc['lastMonthAvgPriceTarget']

            current_price = pd.DataFrame(self.request('v3', 'quote-short')).iloc['price']

            print(f'Estimated upside due to option pricing for the next 12 months is {option_upside}')
            print(f'Average Analyst upside == {(ntm_price_target - current_price) / ntm_price_target}')

            if 'mktCap' in profile.index and fair_value is not None:
                try:
                    # Retrieve the current market cap from the DataFrame
                    current_market_cap = profile.loc['mktCap']

                    # Calculate the percentage difference
                    percentage_difference = (current_market_cap - fair_value) / current_market_cap
                    print("The percentage difference between the current market cap and the fair value is: %.2f%%" % (percentage_difference * 100))
                except:
                    pass
            
            return fair_value

        except Exception as e:
            print(f"Error occurred while processing {self.symbol}: {e}")
            return None
        
        
if __name__ == "__main__":

    start_time = time.time()

    load_dotenv() 

    ticker = 'AMZN' 

    fmp_api_key = os.getenv('FMP_API_KEY')
    fred_api_key = os.getenv('FRED_API_KEY')

    if fmp_api_key is None:
        raise ValueError("API_KEY not found in environment variables. Please set it in your .env file. You can buy a subscription at Financial Modeling Prep: https://site.financialmodelingprep.com/developer/docs/pricing")
    
    DCF = DiscountedCashFlows(ticker, fmp_api_key, fred_api_key) 

    fair_value = DiscountedCashFlows.fair_value

    print(f'Based on estimates, the fair value for {ticker} = {fair_value}')

    print(f'Program took {ticker} = {fair_value}')
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Program took: {elapsed_time} seconds to complete")
