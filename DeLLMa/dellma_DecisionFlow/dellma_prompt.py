agriculture_extraction = """
You are an expert in information extraction and summarization. Your task is to analyze the given text and extract all key pieces of information that might be valuable while ensuring that every extracted sentence explicitly includes the subject from the provided variables instead of using pronouns.

For example, if the variables are two firuts, you should extract information about its Price, Yield and Influencing Factors that will cause its price and yield to change:

# Variables:
Fruit 1: apple
Fruit 2: avocado

# Input text:
"Below is an agriculture report published by the USDA. It gives an overview of the fruit and nut market in the United States, with an additional focus on information pertaining to apple, avocado.

Market Overview: the usda report indicates a general increase in u.s. production of major noncitrus fruits for 2021, with apples, grapes, peaches, cranberries, and sweet and tart cherries seeing a rise in production, while pear production is forecasted to decline. the impact of extreme weather events and california's ongoing drought on crop yields is uncertain. fruit and tree nut grower price indices remain high, with fluctuations throughout 2021. the consumer price index for fresh fruit also increased, suggesting higher retail prices. the northwest heat dome has introduced production uncertainty, particularly for tree fruits. the u.s. citrus season ended with declines in all commodities except california tangerines, and citrus prices are higher. tree nut supplies are forecasted to be down from the previous year's record, with smaller almond and walnut crops expected to increase grower prices. factors such as weather conditions, supply chain issues, and demand are influencing the market.

- apple:
    - Product Summary: apple production is forecasted to be up 3 percent from 2020/21 but down 5 percent from 2019/20. washington state's crop is expected to be larger, but there is concern over heat damage. export markets may remain sluggish due to high tariffs and shipping challenges, potentially pushing more apples into the domestic market and lowering prices. processing prices may rise due to declines in new york and michigan, which account for a significant portion of processed apples.
    - California Price and Yield Statistics: the average apple yield is 19,000 LB / ACRE and the average price per unit is 0.244 $ / LB.
- avocado:
    - Product Summary: california avocado production has decreased, with wildfires and water restrictions impacting yields. however, u.s. avocado consumption has increased significantly, with imports from mexico and peru growing substantially. mexico dominates the u.s. avocado market, with imports peaking from may through july. peruvian imports compete during the summer months, traditionally a period of lower mexican imports.
    - California Price and Yield Statistics: the average avocado yield is 2.87 TONS / ACRE and the average price per unit is 2,430 $ / TON.

I'm a farmer in California planning what fruit to plant next year. I would like to maximize my profit with '10' acres of land."

# Correct Extracted Information:
```json
{{
    "fruits":[
        {{
            "Name": "Apple",
            "Price": "The average price per unit is 0.244 $/LB.",
            "Yield": "The average yield is 19,000 LB/ACRE.",
            "Influencing Factors": [
                "Production is forecasted to be up 3% from 2020/21 but down 5% from 2019/20.",
                "Washington state's crop is expected to be larger, but heat damage is a concern.",
                "Export markets may remain sluggish due to high tariffs and shipping challenges, potentially pushing more apples into the domestic market and lowering prices.",
                "Processing prices may rise due to declines in New York and Michigan, which account for a significant portion of processed apples."
            ]
        }},
        {{
            "Name": "Avocado",
            "Price": "The average price per unit is 2,430 $/TON.",
            "Yield": "The average yield is 2.87 TONS/ACRE.",
            "Influencing Factors":[
                "California avocado production has decreased due to wildfires and water restrictions.",
                "U.S. avocado consumption has increased significantly, with imports from Mexico and Peru growing substantially.",
                "Mexico dominates the U.S. avocado market, with imports peaking from May through July.",
                "Peruvian imports compete during the summer months, traditionally a period of lower Mexican imports."
            ]
        }}
    ]
}}
```

Now, apply the same extraction process to the text provided below and output only the extracted information. Ensure that every sentence includes the certain subject from the variables when applicable.

The text are given below:
<text>
{text}
</text>

The variables are given below:
<variables>
{variable}
</variables>

Output format:
```json
{{<Your answer here in json format>}}
```
"""

stock_extraction = """
You are tasked with extracting financial factors from the provided stock data.
For each factor, you must output:

- value: the numerical result (Current Price retains full precision; other factors in percentages, rounded to two decimal places).

- analysis: a multi-sentence paragraph including:

- A description of the current state

- A brief review of the historical trend

- A prediction of likely future movement based solely on this individual factor.

For example, if the variables are two stocks, you should extract:
- Current Price: The latest available monthly closing price.
- Recent 3-month Return: The percentage return from 3 months ago to the current price.
- Recent 6-month Return: The percentage return from 6 months ago to the current price.
- Annualized Volatility: The standard deviation of monthly returns over the past 12 months, annualized by multiplying by \sqrt(12).
- CAGR (Compound Annual Growth Rate): The annual growth rate of the stock assuming compounding, from the first to the last price.
- Year-over-Year Return: The percentage return from the same month last year to the current price.
- Technical Rebound Analysis: Indicate "Yes" or "No" whether the stock is undergoing a technical rebound.

All values should be given in percentage (e.g., 12.34), rounded to two decimal places, except for Current Price which should retain full precision.

# Input text:
Below are the information about stock AMD (i.e. Advanced Micro Devices). Units are in dollars per share.
Current Price: 119.88.
Historical Prices:
	2021-12: 143.49
	2022-01: 126.84
	2022-02: 119.63
	2022-03: 112.68
	2022-04: 95.80
	2022-05: 94.27
	2022-06: 90.85
	2022-07: 82.90
	2022-08: 96.37
	2022-09: 74.99
	2022-10: 60.32
	2022-11: 69.61
	2022-12: 68.09
	2023-01: 70.27
	2023-02: 82.07
	2023-03: 90.47
	2023-04: 90.81
	2023-05: 102.22
	2023-06: 117.79
	2023-07: 113.69
	2023-08: 108.82
	2023-09: 103.11
	2023-10: 102.56
	2023-11: 117.59
    
# Correct Extracted Information:
"AMD": 
"Current Price": 
	"value": 119.88,
	"analysis": "The current price of AMD is 119.88, which shows a strong recovery compared to previous months. After a period of consolidation around the 100-110 range, the stock has managed to break higher. Based on the current price trend alone, it is likely that AMD may continue to seek higher levels in the short term unless major resistance is encountered."
"Recent 3-month Return (%)": 
	"value": 16.28,
	"analysis": "Over the past 3 months, AMD's stock price has increased by 16.28%. This indicates a strong short-term upward momentum following previous market weakness. The positive three-month return suggests that investor sentiment has improved. If the trend persists, further moderate gains could be expected in the next few months, although short-term pullbacks cannot be ruled out."
"Recent 6-month Return (%)":
	"value": 2.48,
	"analysis": "In the last 6 months, AMD has recorded a modest 2.48% return. This suggests a period of consolidation with slight upward bias. Although the medium-term growth is not particularly strong, stability after prior declines is a positive signal. Based purely on this 6-month return, future movements are likely to be range-bound unless a catalyst triggers a breakout."
"Annualized Volatility (%)": 
	"value": 26.41,
	"analysis": "The annualized volatility for AMD over the past year stands at 26.41%. This level of volatility indicates moderate fluctuations, neither extremely stable nor highly volatile. Historically, such volatility could suggest that while upward moves are possible, investors should expect intermittent sharp price swings. Future short-term movements are likely to remain moderately volatile."
"CAGR (%)": 
	"value": -6.44,
	"analysis": "AMD's compound annual growth rate (CAGR) over the analyzed period is -6.44%. This negative growth rate reflects a decline in stock value when compounding is considered. It shows that despite recent rebounds, longer-term trends have been unfavorable. Based solely on the negative CAGR, the stock might face continued resistance in establishing a strong long-term uptrend unless fundamentals significantly improve."
"Year-over-Year Return (%)":
	"value": 76.02,
	"analysis": "Compared to the same month last year, AMD's price has risen by 76.02%. This impressive year-over-year gain suggests that the company has recovered strongly over the past year. High YoY returns often indicate momentum that could carry forward, though such sharp gains sometimes precede profit-taking. Based on this YoY return, short-term strength could continue, but investors should watch for signs of exhaustion."
"Technical Rebound Analysis":
	"value": "Yes",
	"analysis": "AMD is currently undergoing a technical rebound, as evident by the sharp rise from prior low points. After bottoming out, the stock has staged a sustained recovery, reclaiming important technical levels. Given the rebound, continued upward movement is possible, though future gains may slow as the stock approaches resistance zones."

    
Your turn:
{text}
"""

math_attribute = """
You will receive a fruit name and an influencing factor. Your task is to determine how this influencing factor will affect the fruit's price, yield, or both in the next year.

# Instructions:
1. Carefully read the influencing factor and identify whether it relates to production (yield), market conditions (price), or both.
2. If the factor mentions production levels, weather, or crop health, it likely affects yield.
3. If the factor mentions supply shortages, demand changes, or trade policies, it likely affects price.
4. If the factor impacts both production and market conditions, choose both.
5. Provide the output in the specified JSON format.

For example:
Fruit Name: Apple
Influencing Factors: Production is forecasted to be up 3% from 2020/21 but down 5% from 2019/20.

Correct output:
```json
{{
    "Fruit Name": "Apple",
    "Influencing Factors": "Production is forecasted to be up 3% from 2020/21 but down 5% from 2019/20.",
    "Attribute": "yield"
}}
```

Your Turn:
Input:
Fruit Name: {name}
Influencing Factors: {factor}

Output format:
```json
{{
    "Fruit Name": "<fruit name>",
    "Influencing Factors": "<influencing factors>",
    "Attribute": "<yield or price or both>",
    "Explanation": "<Your explanation why this factor will affect the attribute you choose>"
}}
```
"""

math_attribute_error = """
You will receive a fruit name and an influencing factor. Your task is to determine how this influencing factor will affect the fruit's price, yield, or both in the next year.

# Instructions:
1. Carefully read the influencing factor and identify whether it relates to production (yield), market conditions (price), or both.
2. If the factor mentions production levels, weather, or crop health, it likely affects yield.
3. If the factor mentions supply shortages, demand changes, or trade policies, it likely affects price.
4. If the factor impacts both production and market conditions, choose both.
5. Provide the output in the specified JSON format.

For example:
Fruit Name: Apple
Influencing Factors: Production is forecasted to be up 3% from 2020/21 but down 5% from 2019/20.

Correct output:
```json
{{
    "Fruit Name": "Apple",
    "Influencing Factors": "Production is forecasted to be up 3% from 2020/21 but down 5% from 2019/20.",
    "Attribute": "yield"
}}
```

Your Turn:
Input:
{information}

Output format:
```json
{{
    "Fruit Name": "<fruit name>",
    "Influencing Factors": "<influencing factors>",
    "Attribute": "<yield or price or both>",
    "Explanation": "<Your explanation why this factor will affect the attribute you choose>"
}}
```
"""

math_filter = """
Task: Assess the probability that a given factor will impact a specified attribute, and estimate up to one of the most likely impact scenarios (e.g., increase by 3%, decrease by 5%, no change). For each scenario, categorize the likelihood into one of six levels: "very likely", "likely", "somewhat likely", "somewhat unlikely", "unlikely", or "very unlikely".

# Instructions:
1. Identify the factor and the attribute it may affect.

- Clearly state the factor (e.g., production change, weather conditions, demand shift) and the attribute it might influence (e.g., price, yield or both).

2. Assess up to three possible impact outcomes.
Each outcome should include:

- A likelihood category based on the following scale:

1. "very likely": Almost certain impact
2. "likely": High probability
3. "somewhat likely": Moderate probability
4. "somewhat unlikely": Low probability
5. "unlikely": Very low probability
6. "very unlikely": Almost no impact expected

- An estimated impact magnitude (e.g., "3% increase", "5% decrease", "no change").

- A concise explanation of why this impact is possible and how the likelihood was assessed.

3. Output format:
{{
  "Factor": "<Name of impact factor>",
  "Value": [
    {{
      "Likelihood": "<likelihood category>",
      "ImpactMagnitude": "<estimated change (e.g., '3% increase', 'no change', '5% decrease')>",
      "Explanation": "<Concise explanation>"
    }}
  ]
}}

# For example:

Input:
Name: Apple
Price: The average price per unit is 0.244 $/LB.
Yield: The average yield is 19,000 LB/ACRE.
Factor: Apple production is forecasted to be up 3% from 2020/21 but down 5% from 2019/20.
Attribute: yield

Output:
```json
{{
  "Factor": "Apple production is forecasted to be up 3% from 2020/21 but down 5% from 2019/20.",
  "Value": [
    {{
      "Likelihood": "likely",
      "ImpactMagnitude": "3% increase",
      "Explanation": "The production forecast shows a 3% increase from 2020/21, which is likely to correspond to a similar rise in yield given that production and yield are directly related."
    }}
  ]
}}
```

# Your Turn:
Name: {fruit_name}
Price: {fruit_price}
Yield: {fruit_yield}
Factor: {fruit_factor}
Attribute: {fruit_attribute}
"""

math_express = """
You are an agricultural economist tasked with estimating the price and yield of apples for the next year based on the following instruction:

# Instructions:

1. Likelihood Encoding: Convert the "Likelihood" of each change into a probability value between 0 and 1 using the following scale:

- "very likely" → 0.9

- "likely" → 0.7

- "somewhat likely" → 0.5

- "somewhat unlikely" → 0.3

- "unlikely" → 0.1

- "very unlikely" → 0.0

2. Impact Calculation: For each change, calculate the expected impact on the attribute (price or yield) by multiplying the impact magnitude by the likelihood probability.

3. Aggregate Impact: Sum the expected impacts for each attribute (price and yield) separately.

4. Final Estimation: Adjust the current price and yield by the aggregated impacts to estimate the price and yield for the next year.

# For example:
Name: Apple

Current Price: The average price per unit is 0.244 $/LB.

Current Yield: The average yield is 19,000 LB/ACRE.

Changes:

{{'Attribute': 'yield', 'ImpactMagnitude': 'a 1.5% decrease', 'Likelihood': 'somewhat likely'}}

{{'Attribute': 'yield', 'ImpactMagnitude': '2% decrease', 'Likelihood': 'somewhat likely'}}

{{'Attribute': 'price', 'ImpactMagnitude': '5-7% decrease', 'Likelihood': 'somewhat likely'}}

{{'Attribute': 'price', 'ImpactMagnitude': '5-7% increase', 'Likelihood': 'somewhat likely'}}

Example output:
```json
{{
    "Estimated Price": "0.235 $/LB (a 3.7% decrease from the current price)",
    "Estimated Yield": "18,600 LB/ACRE (a 2.1% decrease from the current yield)",
    "Explanation": "Price Impact: The 5-7% decrease in price with a 50% likelihood contributes an expected decrease of 2.5-3.5%. The 5-7% increase in price with a 50% likelihood contributes an expected increase of 2.5-3.5%. The net impact on price is a slight decrease of 0-1%, resulting in an estimated price of 0.235 $/LB. Yield Impact: The 1.5% decrease in yield with a 50% likelihood contributes an expected decrease of 0.75%. The 2% decrease in yield with a 50% likelihood contributes an expected decrease of 1%. The net impact on yield is a decrease of 1.75%, resulting in an estimated yield of 18,600 LB/ACRE."
}}
```

# Your Turn:
Name: {fruit_name}

Current Price: {fruit_price}

Current Yield: {fruit_yield}

Changes: {fruit_change}

Output format:
```json
{{
    "Estimated Price": "<estimated next year's price>",
    "Estimated Yield": "<estimated next year's yield>",
    "Explanation": "<explanation for the price and yield using Mathematical formula>"
}}
```
"""

math_reason = """
{objective}

{actions}

Based on the given information, which is an estimation for next year, choose the correct answer.
{structure}

{preference}
"""

agriculture_distribute = """
For each relevant impact factor (e.g., drought, demand surge, import competition), identify which output variable it affects (price or yield) and choose an appropriate probability distribution to model the uncertainty.
Examples of distributions you can use include:

Normal: when changes cluster around a central value with known variance

Output a list of distributions like:
```json
[
{{
    "Name": "Heatwave impact on Apple Yield",
    "Affects": "yield",
    "Distribution": "Normal(-0.2, 0.01)"
}},
{{
    "Name": "Export barriers on Apple Price",
    "Affects": "price",
    "Distribution": "Normal(-0.05, 0.03)"
}}
]
```
Your turn:
{information}

Output Format:
```json
{{
    "Name":
    "Affects":
    "Distribution":
}}
```
"""

stocks_distribute = """
Based on the following historical data for the given stock, estimate its likely closing price at the end of the next month (in USD). Analysis every stock and output the whole results for each stock.

First, rank all available factors by their importance for forecasting future stock price:

Classify each factor as:

High Importance (must be heavily considered; major impact on forecast)

Medium Importance (helpful reference; supportive but not decisive)

Low Importance (can be ignored or minimally referenced)

For example:

"AMD": 
"Current Price": 
	"value": 119.88,
	"analysis": "The current price of AMD is 119.88, which shows a strong recovery compared to previous months. After a period of consolidation around the 100-110 range, the stock has managed to break higher. Based on the current price trend alone, it is likely that AMD may continue to seek higher levels in the short term unless major resistance is encountered."
"Recent 3-month Return (%)": 
	"value": 16.28,
	"analysis": "Over the past 3 months, AMD's stock price has increased by 16.28%. This indicates a strong short-term upward momentum following previous market weakness. The positive three-month return suggests that investor sentiment has improved. If the trend persists, further moderate gains could be expected in the next few months, although short-term pullbacks cannot be ruled out."
"Recent 6-month Return (%)":
	"value": 2.48,
	"analysis": "In the last 6 months, AMD has recorded a modest 2.48% return. This suggests a period of consolidation with slight upward bias. Although the medium-term growth is not particularly strong, stability after prior declines is a positive signal. Based purely on this 6-month return, future movements are likely to be range-bound unless a catalyst triggers a breakout."
"Annualized Volatility (%)": 
	"value": 26.41,
	"analysis": "The annualized volatility for AMD over the past year stands at 26.41%. This level of volatility indicates moderate fluctuations, neither extremely stable nor highly volatile. Historically, such volatility could suggest that while upward moves are possible, investors should expect intermittent sharp price swings. Future short-term movements are likely to remain moderately volatile."
"CAGR (%)": 
	"value": -6.44,
	"analysis": "AMD's compound annual growth rate (CAGR) over the analyzed period is -6.44%. This negative growth rate reflects a decline in stock value when compounding is considered. It shows that despite recent rebounds, longer-term trends have been unfavorable. Based solely on the negative CAGR, the stock might face continued resistance in establishing a strong long-term uptrend unless fundamentals significantly improve."
"Year-over-Year Return (%)":
	"value": 76.02,
	"analysis": "Compared to the same month last year, AMD's price has risen by 76.02%. This impressive year-over-year gain suggests that the company has recovered strongly over the past year. High YoY returns often indicate momentum that could carry forward, though such sharp gains sometimes precede profit-taking. Based on this YoY return, short-term strength could continue, but investors should watch for signs of exhaustion."
"Technical Rebound Analysis":
	"value": "Yes",
	"analysis": "AMD is currently undergoing a technical rebound, as evident by the sharp rise from prior low points. After bottoming out, the stock has staged a sustained recovery, reclaiming important technical levels. Given the rebound, continued upward movement is possible, though future gains may slow as the stock approaches resistance zones."

Take into account the recent return trends, the influence of volatility on potential price fluctuations, and any other relevant financial metrics. Indicate whether your estimate is conservative, optimistic, or neutral, and justify your assumptions.

If there are more than one stock, then output them all.

Output a list of distributions like:
```json
{{
  "AMD": {{
    "Price": 140.50,
    "Reason": "The price prediction for AMD is based on a structured evaluation of multiple factors with different levels of importance. \n\nFirst, the most influential factors are the Recent 3-month Return (16.28%), the Year-over-Year Return (76.02%), and the confirmation of a Technical Rebound. The strong 3-month return shows robust short-term momentum, suggesting improving investor sentiment and continued interest in the stock. The very high Year-over-Year return further confirms a sustained recovery trajectory over the past 12 months, which often signals ongoing bullish momentum. The Technical Rebound indicates that AMD has moved up sharply from recent lows, typically implying more room for upward movement unless significant resistance levels are met.\n\nSecondly, factors of medium importance include the Current Price (119.88 USD) and Annualized Volatility (26.41%). The current price has broken out of previous consolidation levels (100-110 range), suggesting technical strength. Meanwhile, the moderate volatility level indicates that price swings are expected but not extreme, supporting a reasonable expectation of gradual gains with intermittent pullbacks.\n\nThirdly, factors of low importance for this short-term forecast are the Recent 6-month Return (2.48%) and the negative Compound Annual Growth Rate (CAGR) of -6.44%. The weak 6-month return suggests some previous stagnation, but it has been recently overcome by stronger short-term momentum. The negative long-term CAGR reflects historical weakness but is not considered critical for a one-month outlook.\n\nConsidering all these factors, the forecast is optimistic. The strong recent momentum (both 3-month and YoY returns) and technical recovery outweigh concerns from medium-term consolidation and long-term historical decline. As a result, a moderately optimistic forecast places AMD at approximately 140.50 USD at the end of next month, assuming no major market disruptions occur."
  }},
  ...
}}
```

Your turn:
{information}

Output Format:
```json
{{
    "<Stock1 Name>": {{
    "Price": 
    "Reason": 
    }},
    "<Stock2 Name>": {{
    "Price": 
    "Reason": 
    }},
}}
```
"""

math_mean = """{information}
{distribute}
Step 1:
For each distribution defined above, calculate:

Expected value (mean) of the impact on yield or price

Variance to reflect uncertainty

Step 2:
Use the objective function for profit:

Profit=(Base Yield*(1+Yield Mean))*(Base Price*(1+Price Mean))*Acreage-Cost
Perform this calculation for each decision variable. Include:

Base price and yield

Estimated adjustments above

Cost (optional, if known or estimated)

Then compute the expected profit numerically, e.g.:
```json
{{
"Apple Profit": 46360,
"Avocado Profit": 69741
}}
```
"""
