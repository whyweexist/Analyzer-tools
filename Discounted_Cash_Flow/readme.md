# Discounted Cash Flows Automated

<p id="description">
    This tool is designed to calculate the perceived value of a company based strictly on quantitative figures. 
</p>

- Removing emotions from intrinsic value
- Ability to change and identify trends within the data
- Capitalize on short term polarizing events that misprice quality assets

  

> [!NOTE]
> This module is still evolving and may change. Feel free to build and experiment, but please don't rely on its stability just yet!


**Key Features and Scope**

This project automates Discounted Cash Flow (DCF) calculations to estimate company value, using analyst EBIT and revenue estimates along with perpetual growth rates like 2-year government bond yields. 

It allows for customizable inputs, such as reinvestment and tax rates, and integrates detailed financial data from various APIs, including industry metrics and macroeconomic indicators.

**Subjectivity in DCF Analysis**

Discounted Cash Flow (DCF) analysis is inherently subjective because it relies on factors like market sentiment, competitive landscape, and the quality of management, all of which can affect valuations. 

Economic changes and technological progress also make it challenging to predict future performance accurately. On a short-term basis, investor sentiment and large buy or sell orders can sway stock prices, presenting opportunities for strategic trades. This program aims to offer a clearer, more rational approach to understanding the often chaotic stock market.


> **Tip** To use this program, you will need to create an account and subscribe to the APIs listed below.



 [Financial Modeling Prep](https://site.financialmodelingprep.com/developer/docs/pricing)
 
 [St. Louis Fed](https://fredaccount.stlouisfed.org/apikeys)




<h2>🛠️ Installation Steps:</h2>

<p>1. Clone the Repository:</p>

```
git clone https://github.com/realjake/Automated-DCF.git
```

<p>Navigate into the cloned repo</p>

```
cd AutomatedDCF
```


<p>2. Install Required Dependencies:</p>

```
pip install -r requirements.txt
```


<p>3. Edit .env file and add your Financial Modeling Prep API key:</p>

```
--> AutomatedDCF / .env / FMP_API_KEY=XXXXXXXX
```
