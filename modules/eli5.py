"""
ELI5 (Explain Like I'm 5) Glossary for Stock Analysis Terms

A plain-English encyclopedia powering the "ELI5" tab in the Streamlit app.
Every term used in the analysis dashboard has a conversational explanation,
real-world analogies, and clear good/bad indicators.
"""

GLOSSARY = {
    # ========== TECHNICAL ANALYSIS ==========
    "RSI": {
        "full_name": "Relative Strength Index",
        "category": "Technical",
        "eli5": "A 'tired-o-meter' for a stock. If it's been running up really fast, it gets tired (overbought) and might rest. If it's been falling hard, it gets oversold and might bounce back.",
        "what_it_means": "RSI measures how fast a stock has gone up or down recently on a scale of 0-100. Readings above 70 mean it's risen a lot (possibly too much), below 30 means it's fallen a lot (possibly too much). It helps spot when a stock might be due for a reversal or pause.",
        "good_sign": "RSI between 40-60 (healthy momentum) or bouncing up from below 30 (potential recovery).",
        "bad_sign": "RSI stuck above 70 (overbought, pullback risk) or below 30 consistently (severe weakness).",
        "emoji": "📈",
    },
    "MACD": {
        "full_name": "Moving Average Convergence Divergence",
        "category": "Technical",
        "eli5": "Two lines that hug each other when a stock is calm, and separate when it's getting excited. When they cross over, it's like the stock changing direction.",
        "what_it_means": "MACD tracks the relationship between two moving averages (fast and slow). When the fast line crosses above the slow one, it's a bullish signal. When it crosses below, it's bearish. The wider they spread, the stronger the momentum.",
        "good_sign": "MACD line above the signal line and both rising; or MACD crossing up through the signal line (bullish crossover).",
        "bad_sign": "MACD line below the signal line and both falling; or MACD crossing down (bearish crossover).",
        "emoji": "🔄",
    },
    "Bollinger Bands": {
        "full_name": "Bollinger Bands",
        "category": "Technical",
        "eli5": "Imagine a train track around a stock's price. The middle is the average, the top and bottom rails are the expected range. When price touches the rails, it's extreme; when the rails squeeze tight, a big move is coming.",
        "what_it_means": "Bollinger Bands are three lines: a moving average in the middle and two standard-deviation bands above and below. They contract when volatility is low and expand when it's high. Prices touching the bands often signal reversals or breakouts.",
        "good_sign": "Price bouncing between the middle and lower band (support); or bands expanding with uptrend (strong momentum).",
        "bad_sign": "Price pinned to the upper band for too long (overbought); or bands squeezing (low volatility before a big move).",
        "emoji": "🎯",
    },
    "SMA 20": {
        "full_name": "Simple Moving Average (20-day)",
        "category": "Technical",
        "eli5": "The average price over the last 20 days. It's like a 3-week heartbeat for the stock — shows if the stock is generally going up, down, or sideways.",
        "what_it_means": "The 20-day SMA smooths out daily noise and shows recent momentum. It's used as a short-term trend indicator and often acts as a dynamic support or resistance level.",
        "good_sign": "Price above the 20-day SMA (short-term uptrend); SMA angled upward.",
        "bad_sign": "Price below the 20-day SMA (short-term downtrend); SMA angled downward.",
        "emoji": "📊",
    },
    "SMA 50": {
        "full_name": "Simple Moving Average (50-day)",
        "category": "Technical",
        "eli5": "The average price over the last 50 days — about 2.5 months. It's the medium-term trend line. If the short-term trend crosses above this, the stock is getting legs.",
        "what_it_means": "The 50-day SMA shows the medium-term direction. When the 20-day SMA crosses above it (bullish crossover), it signals a strengthening uptrend. When it crosses below (bearish), downtrend is building.",
        "good_sign": "Price above the 50-day SMA; 20-day SMA above 50-day SMA; 50-day SMA rising.",
        "bad_sign": "Price below the 50-day SMA; 20-day SMA below 50-day SMA; 50-day SMA falling.",
        "emoji": "📈",
    },
    "SMA Crossover": {
        "full_name": "SMA Crossover (20/50)",
        "category": "Technical",
        "eli5": "When the fast 20-day line crosses above the slower 50-day line, it's like a sprinter overtaking a marathoner — uptrend! When it crosses below, downtrend ahead.",
        "what_it_means": "A golden cross (20-day crosses above 50-day) is one of the most popular bullish signals. A death cross (20-day crosses below 50-day) is bearish. These crossovers often mark regime changes in the stock's momentum.",
        "good_sign": "20-day SMA crossing above 50-day SMA (golden cross); both SMAs rising afterward.",
        "bad_sign": "20-day SMA crossing below 50-day SMA (death cross); both SMAs falling afterward.",
        "emoji": "🔀",
    },
    "SMA Crossover Backtest": {
        "full_name": "SMA Crossover Historical Win Rate",
        "category": "Technical",
        "eli5": "If you had bought every time the 20-day line crossed above the 50-day line in the past, how often would you have made money? That's the backtest win rate.",
        "what_it_means": "A backtest score shows the historical success rate of the SMA crossover strategy on this specific stock. For example, a 65% win rate means 65% of past buy signals were profitable. Higher is better, but don't rely on it alone.",
        "good_sign": "Backtest win rate above 55-60% (strategy historically worked on this stock).",
        "bad_sign": "Backtest win rate below 50% (strategy hasn't worked well historically).",
        "emoji": "📉",
    },
    "Volume": {
        "full_name": "Trading Volume",
        "category": "Technical",
        "eli5": "How many shares were bought and sold today. High volume = lots of people agree the price is fair. Low volume = few people trading = less conviction.",
        "what_it_means": "Volume confirms price movement. A big price jump on high volume is more convincing than on low volume (easier to manipulate). Volume spikes often precede big moves or reversals.",
        "good_sign": "Price rising on increasing volume (strong uptrend); volume spike on breakout.",
        "bad_sign": "Price rising on decreasing volume (weak uptrend, may not last); volume drying up during a rally.",
        "emoji": "📊",
    },
    "Candlestick Chart": {
        "full_name": "Candlestick Chart",
        "category": "Technical",
        "eli5": "A chart where each candle shows a day's action: the green/red body is open-to-close, and the wicks show the high and low. Green = up day, red = down day.",
        "what_it_means": "Candlestick charts show detailed price action in a compact form. The body (open-to-close) reveals who controlled the day. Long wicks show rejections (price tried to go up/down but got pushed back).",
        "good_sign": "Big green candles with small lower wicks (buyers in control); candles closing at highs.",
        "bad_sign": "Big red candles with small upper wicks (sellers in control); candles closing at lows.",
        "emoji": "🕯️",
    },
    "Support Level": {
        "full_name": "Support Level",
        "category": "Technical",
        "eli5": "A price level that acts like a floor. Every time the stock falls to this level, buyers jump in and push it back up, like a bouncy ball.",
        "what_it_means": "Support is a price where buying pressure historically comes in, preventing further declines. It's often based on previous lows, round numbers, or technical indicators. When support breaks, it's a bearish signal.",
        "good_sign": "Price bouncing up off support level; support holding through multiple tests.",
        "bad_sign": "Price breaking below support; volume surging through support (conviction of sellers).",
        "emoji": "🔻",
    },
    "Resistance Level": {
        "full_name": "Resistance Level",
        "category": "Technical",
        "eli5": "A price level that acts like a ceiling. Every time the stock rises to this level, sellers show up and push it back down, like a spring.",
        "what_it_means": "Resistance is a price where selling pressure historically comes in, preventing further gains. It's often based on previous highs, round numbers, or technical indicators. Breaking above resistance is bullish.",
        "good_sign": "Price breaking above resistance on high volume; resistance becoming support after breakout.",
        "bad_sign": "Price bouncing down off resistance repeatedly; failed breakout attempts.",
        "emoji": "🔺",
    },
    "Pivot Point": {
        "full_name": "Pivot Point",
        "category": "Technical",
        "eli5": "A calculated price level based on yesterday's high, low, and close. It's like the stock's expected balance point for the day — the level around which it might rotate.",
        "what_it_means": "Pivot points are support/resistance levels derived from previous price data. Traders use them as turning points. The pivot itself is the main level; S1/S2 are support levels, R1/R2 are resistance levels.",
        "good_sign": "Price bouncing off pivot point levels; clean reversals at S1/R1.",
        "bad_sign": "Price ignoring pivot points; price gapping far from pivots.",
        "emoji": "🎪",
    },

    # ========== FUNDAMENTAL ANALYSIS ==========
    "P/E Ratio": {
        "full_name": "Price-to-Earnings Ratio",
        "category": "Valuation",
        "eli5": "Imagine buying a lemonade stand that makes $1,000/year. If you pay $15,000, your P/E is 15. High P/E = you're paying a lot, betting it'll grow fast. Low P/E = cheap, but maybe cheap for a reason.",
        "what_it_means": "P/E is the stock price divided by annual earnings per share. It tells you how many years of earnings you're paying for. A low P/E might mean undervalued or declining business. A high P/E might mean overvalued or fast-growing.",
        "good_sign": "P/E lower than industry average and S&P 500 (around 20); P/E low but earnings growing fast.",
        "bad_sign": "P/E much higher than peers or market; P/E elevated while earnings shrinking.",
        "emoji": "💰",
    },
    "Forward P/E": {
        "full_name": "Forward Price-to-Earnings Ratio",
        "category": "Valuation",
        "eli5": "Same idea as P/E, but using what analysts predict the company will earn next year instead of what it earned last year. Future-focused version.",
        "what_it_means": "Forward P/E divides current stock price by projected next-year earnings. It's more forward-looking than trailing P/E. A falling forward P/E as earnings grow is a good sign (stock getting cheaper).",
        "good_sign": "Forward P/E lower than trailing P/E (market expects earnings growth); forward P/E lower than sector average.",
        "bad_sign": "Forward P/E higher than trailing P/E (earnings expected to decline); forward P/E elevated vs peers.",
        "emoji": "🔮",
    },
    "EPS": {
        "full_name": "Earnings Per Share",
        "category": "Fundamental",
        "eli5": "Imagine a company makes $100 million profit and has 100 million shares. Each share 'owns' $1 of that profit. That's the EPS.",
        "what_it_means": "EPS is net income divided by shares outstanding. It shows how much profit each share of stock represents. Growing EPS is bullish; shrinking EPS is bearish.",
        "good_sign": "EPS growing year-over-year; EPS beating analyst expectations; EPS growing faster than revenue.",
        "bad_sign": "EPS declining or flat; EPS falling short of expectations; EPS falling while revenue grows (margin compression).",
        "emoji": "💵",
    },
    "Revenue": {
        "full_name": "Total Revenue",
        "category": "Fundamental",
        "eli5": "All the money the company made from selling products/services before paying any expenses. It's the top line of the income statement.",
        "what_it_means": "Revenue is the total sales the company generated. It doesn't account for expenses, so it's the starting point. Revenue growth shows demand for the product, but doesn't guarantee profitability.",
        "good_sign": "Revenue growing year-over-year consistently; revenue beating expectations; revenue growth accelerating.",
        "bad_sign": "Revenue flat or declining; revenue missing expectations; revenue growth slowing.",
        "emoji": "📈",
    },
    "Revenue Growth": {
        "full_name": "Revenue Growth Rate",
        "category": "Fundamental",
        "eli5": "How fast the company's sales are growing. A company growing 30% per year is expanding much faster than one growing 5%.",
        "what_it_means": "Revenue growth (usually year-over-year percentage) shows how quickly the business is expanding. High growth is bullish but may not be sustainable. Compare to peer and industry averages.",
        "good_sign": "Double-digit revenue growth; accelerating growth quarter-over-quarter; growth above industry average.",
        "bad_sign": "Revenue growth declining; growth below industry average; growth turning negative.",
        "emoji": "📊",
    },
    "Profit Margin": {
        "full_name": "Net Profit Margin",
        "category": "Fundamental",
        "eli5": "Out of every $1 the company makes in sales, how much is actual profit after all expenses? A 20% margin means $0.20 profit per $1 of sales.",
        "what_it_means": "Profit margin (net income / revenue) shows operational efficiency. Higher margins mean the company keeps more of each sale as profit. Expanding margins are bullish; contracting margins are bearish.",
        "good_sign": "Profit margin above industry average; margins expanding year-over-year; high margins (10-20%+ for most industries).",
        "bad_sign": "Profit margin shrinking; margins below industry average; negative profit margin (company losing money).",
        "emoji": "💯",
    },
    "ROE": {
        "full_name": "Return on Equity",
        "category": "Fundamental",
        "eli5": "If you invested $1,000 in the company, how much annual profit would it generate for you? ROE of 15% means $150/year on $1,000. Higher is better.",
        "what_it_means": "ROE (net income / shareholders' equity) measures how efficiently the company uses shareholder money to generate profits. A high ROE (above 15%) suggests good management and competitive advantage.",
        "good_sign": "ROE above 15%; ROE higher than peers; ROE stable or increasing.",
        "bad_sign": "ROE below 10%; ROE declining; ROE far below industry average.",
        "emoji": "📈",
    },
    "Debt/Equity": {
        "full_name": "Debt-to-Equity Ratio",
        "category": "Fundamental",
        "eli5": "How much the company owes compared to what shareholders own. A ratio of 1.0 means the company owes as much as shareholders have invested. Higher = more risk.",
        "what_it_means": "Debt/Equity measures financial leverage. A high ratio (>2.0) means the company is heavily indebted and risky in a downturn. Low ratio (<0.5) means conservative capital structure.",
        "good_sign": "Debt/Equity below 1.0; ratio declining over time; company paying down debt.",
        "bad_sign": "Debt/Equity above 2.0; ratio rising; company taking on more debt.",
        "emoji": "⚖️",
    },
    "Free Cash Flow": {
        "full_name": "Free Cash Flow",
        "category": "Fundamental",
        "eli5": "The actual cash a company has left after paying for equipment and operations. It's the money available to pay shareholders, pay down debt, or invest in growth.",
        "what_it_means": "FCF (operating cash flow - capital expenditures) is the cash genuinely available. Unlike earnings, you can't manipulate cash. Positive, growing FCF is a strong sign of financial health.",
        "good_sign": "Positive FCF and growing; FCF higher than net income (quality earnings); FCF turning positive.",
        "bad_sign": "Negative FCF; FCF declining; FCF much lower than net income (profit not converting to cash).",
        "emoji": "💸",
    },
    "FCF Yield": {
        "full_name": "Free Cash Flow Yield",
        "category": "Fundamental",
        "eli5": "The annual free cash flow divided by the company's market value. Like asking: 'If I buy the whole company today, what % cash return does it generate?'",
        "what_it_means": "FCF yield is FCF / market cap. A high FCF yield (>5-10%) suggests the stock is cheap relative to cash generation. Compare to dividend yield and bond yields to assess value.",
        "good_sign": "FCF yield above 5-10%; FCF yield higher than dividend yield (more cash than dividends).",
        "bad_sign": "FCF yield below 2%; negative FCF yield (negative FCF); declining FCF yield.",
        "emoji": "🏦",
    },
    "EV/EBITDA": {
        "full_name": "Enterprise Value-to-EBITDA",
        "category": "Valuation",
        "eli5": "Like P/E, but compares the company's total value (including debt) to earnings before interest, taxes, depreciation. Useful for comparing companies with different debt levels.",
        "what_it_means": "EV/EBITDA is a valuation multiple that ignores capital structure and taxes, making it good for comparing across industries. Lower ratios suggest undervaluation; higher suggest premium pricing.",
        "good_sign": "EV/EBITDA below industry average; EV/EBITDA declining (stock getting cheaper).",
        "bad_sign": "EV/EBITDA far above peers; EV/EBITDA rising; extremely high ratios (>20x).",
        "emoji": "📊",
    },
    "Beta": {
        "full_name": "Beta Coefficient",
        "category": "Risk",
        "eli5": "How much does this stock swing compared to the overall market? Beta of 1.5 means it swings 50% more than the market. High beta = riskier but bigger gains/losses.",
        "what_it_means": "Beta measures volatility vs the S&P 500. Beta = 1.0 matches the market. Beta > 1.0 means more volatile (risky). Beta < 1.0 means less volatile (defensive). Useful for assessing risk tolerance.",
        "good_sign": "Beta between 0.8-1.3 (reasonable risk); low beta (1.0 or below) if risk-averse.",
        "bad_sign": "Beta above 2.0 (very volatile); beta below 0.3 (unusually low, may signal issues).",
        "emoji": "📈",
    },
    "Dividend Yield": {
        "full_name": "Dividend Yield",
        "category": "Fundamental",
        "eli5": "Annual dividend per share divided by stock price. If a stock is $100 and pays $3/year, the yield is 3%. It's your annual cash return just for holding the stock.",
        "what_it_means": "Dividend yield shows the annual cash return you get from holding the stock. Higher yields are attractive but can be unsustainable if too high relative to earnings/FCF.",
        "good_sign": "Dividend yield 2-5% (attractive, sustainable); yield higher than bond rates.",
        "bad_sign": "Dividend yield above 7-10% (unsustainably high, cut risk); yield below 1% (company retains cash).",
        "emoji": "💵",
    },
    "Piotroski F-Score": {
        "full_name": "Piotroski F-Score",
        "category": "Fundamental",
        "eli5": "A 9-point checklist of financial health. It looks at profitability, cash flow, operational efficiency, and leverage. A score of 8-9 is 'financially strong,' 4-5 is 'weak.'",
        "what_it_means": "The F-Score (0-9) is an academic-developed quality signal. It checks: Is the company profitable? Is cash flow positive? Is return improving? Is leverage stable? 8+ is strong, 4 or below is weak.",
        "good_sign": "F-Score of 8-9 (financially strong); F-Score improving year-over-year.",
        "bad_sign": "F-Score of 4 or below (financially weak); F-Score declining.",
        "emoji": "📋",
    },
    "Balance Sheet Trends": {
        "full_name": "Balance Sheet Trends (Debt, Cash, Revenue)",
        "category": "Fundamental",
        "eli5": "Looking at the company's balance sheet year-over-year. Is debt going down? Is cash building up? Is revenue growing? Trends matter more than single snapshots.",
        "what_it_means": "Balance sheet trends reveal financial trajectory. Declining debt, rising cash, and growing assets are bullish. Rising debt, declining cash, and shrinking assets are bearish.",
        "good_sign": "Debt decreasing; cash increasing; assets growing; revenue YoY growth positive.",
        "bad_sign": "Debt increasing sharply; cash declining; assets shrinking; revenue declining.",
        "emoji": "📊",
    },
    "Share Count Trend": {
        "full_name": "Share Count Trend (Buybacks vs Dilution)",
        "category": "Fundamental",
        "eli5": "Is the company buying back its own shares (shrinking the pie but giving you a bigger slice) or issuing new shares (growing the pie but your slice gets smaller)?",
        "what_it_means": "Declining share count (buybacks) can boost EPS without growing earnings — positive but not always a good deal. Rising share count (dilution) suggests fundraising or executive compensation, diluting existing shareholders.",
        "good_sign": "Share count declining (buybacks); share count flat despite growth.",
        "bad_sign": "Share count rising sharply (dilution); aggressive dilution to fund operations.",
        "emoji": "📊",
    },
    "Earnings Surprise": {
        "full_name": "Earnings Surprise (Beat/Miss)",
        "category": "Fundamental",
        "eli5": "Did the company beat, meet, or miss what analysts expected? Beat = good surprise, miss = bad surprise. It matters for near-term momentum.",
        "what_it_means": "Earnings surprise measures actual results vs consensus expectations. Beating EPS expectations is bullish and often triggers rallies. Missing EPS is bearish. Guidance beats/misses matter too.",
        "good_sign": "Beat EPS and revenue expectations; raised forward guidance; multiple beats in a row.",
        "bad_sign": "Missed EPS or revenue expectations; lowered guidance; multiple misses in a row.",
        "emoji": "🎯",
    },

    # ========== MARKET INTELLIGENCE ==========
    "Analyst Consensus": {
        "full_name": "Analyst Consensus Rating",
        "category": "Market Intelligence",
        "eli5": "What do Wall Street analysts think? BUY = bullish, HOLD = neutral, SELL = bearish. It's a crowd vote, but crowds can be wrong.",
        "what_it_means": "Analyst consensus is the average rating across analysts. BUY ratings > HOLD ratings > SELL ratings is bullish. But be aware: analyst ratings are backward-looking and often lag reality.",
        "good_sign": "Strong BUY or BUY consensus (most analysts bullish); consensus upgrading.",
        "bad_sign": "SELL or HOLD consensus; consensus downgrading; more downgrades than upgrades recently.",
        "emoji": "📊",
    },
    "Price Target": {
        "full_name": "Analyst Price Targets",
        "category": "Market Intelligence",
        "eli5": "Where do analysts think the stock will go? If the target is $100 and the stock is at $80, they see $20 upside. Targets are educated guesses, not promises.",
        "what_it_means": "Price targets are analyst forecasts for where the stock should trade. Compare the average target to current price. Upside = (target - current price) / current price. Targets change as outlooks shift.",
        "good_sign": "Price target significantly above current price (implied upside >15%); targets rising.",
        "bad_sign": "Price target below current price (downside expected); targets declining.",
        "emoji": "🎯",
    },
    "Upside %": {
        "full_name": "Upside Potential %",
        "category": "Market Intelligence",
        "eli5": "If the analyst price target comes true, how much could the stock go up? Upside of 20% means the stock could rise 20% from here if the target is right.",
        "what_it_means": "Upside % = (average price target - current price) / current price * 100%. It shows potential return based on analyst consensus. Compare to risk (downside potential) to assess risk/reward.",
        "good_sign": "Upside above 15-20%; upside increasing (targets rising).",
        "bad_sign": "Negative upside (targets below current price); upside declining.",
        "emoji": "📈",
    },
    "Insider Transactions": {
        "full_name": "Insider Buying/Selling",
        "category": "Market Intelligence",
        "eli5": "When the CEO, CFO, or board members buy or sell their own company stock, it's tracked. Insiders buying = they think the stock is cheap. Insiders selling = they might think it's overvalued.",
        "what_it_means": "Insider buying is generally bullish (insiders know the company best). Insider selling is more ambiguous (could be portfolio rebalancing). Look for patterns: multiple insiders buying is very bullish; consistent selling is concerning.",
        "good_sign": "Multiple insiders buying recently; insider buys increasing; insider buying at lower prices.",
        "bad_sign": "Heavy insider selling; CEO/CFO selling large positions; all insiders selling.",
        "emoji": "👔",
    },
    "Institutional Ownership": {
        "full_name": "Institutional Ownership %",
        "category": "Market Intelligence",
        "eli5": "What percentage of the company's shares are owned by big institutional investors like mutual funds, pension funds, and hedge funds? Higher = more 'smart money' invested.",
        "what_it_means": "High institutional ownership (>50%) suggests professional investors believe in the company. Changes in ownership matter: rising ownership is bullish, falling ownership is bearish.",
        "good_sign": "Institutional ownership above 50%; ownership rising; large institutions buying.",
        "bad_sign": "Institutional ownership below 20%; ownership declining; institutions reducing positions.",
        "emoji": "🏛️",
    },
    "Put/Call Ratio": {
        "full_name": "Put/Call Ratio (Options Sentiment)",
        "category": "Market Intelligence",
        "eli5": "A ratio of how many people are betting the stock will fall (puts) vs betting it will rise (calls). High ratio = pessimistic, low ratio = optimistic.",
        "what_it_means": "Put/call ratio measures options market sentiment. A ratio above 1.0 means more puts than calls (bearish). Below 1.0 means more calls (bullish). Extreme ratios sometimes signal reversals.",
        "good_sign": "Put/call ratio below 0.8 (bullish sentiment); ratio declining.",
        "bad_sign": "Put/call ratio above 1.2 (bearish sentiment); ratio rising sharply.",
        "emoji": "📊",
    },
    "Short Interest": {
        "full_name": "Short Interest %",
        "category": "Market Intelligence",
        "eli5": "The percentage of shares that traders have borrowed and sold, betting the price will fall. High short interest means lots of pessimism — or potential for a short squeeze.",
        "what_it_means": "Short interest (% of float shorted) shows negative sentiment. High short interest (>20-30%) is bearish but also creates short-squeeze risk if the stock rallies.",
        "good_sign": "Low short interest (<5%); declining short interest (shorts covering).",
        "bad_sign": "High short interest (>20%); rising short interest.",
        "emoji": "📉",
    },
    "Days to Cover": {
        "full_name": "Days to Cover (Short Interest)",
        "category": "Market Intelligence",
        "eli5": "If all the short sellers tried to buy back their shares at today's volume, how many days would it take? High number = harder to cover = more risk of a squeeze.",
        "what_it_means": "Days to cover = short interest / daily average volume. High DTC (>10 days) means shorts are locked in and a rally could trigger panic-buying. Low DTC (<3 days) means shorts can exit quickly.",
        "good_sign": "Days to cover below 5 (shorts can exit easily); declining.",
        "bad_sign": "Days to cover above 10 (locked-in shorts); rising.",
        "emoji": "⏱️",
    },
    "Short Squeeze": {
        "full_name": "Short Squeeze Risk/Event",
        "category": "Risk",
        "eli5": "You borrowed your friend's bike to sell it cheap, planning to buy a cheaper one later. But the bike got popular and prices shot up. Now you're scrambling to buy it back at a higher price — that panic-buying is a short squeeze.",
        "what_it_means": "A short squeeze happens when heavily-shorted stocks rally, forcing short sellers to buy back (cover) at higher prices, triggering more panic-buying and even faster rallies. It's dramatic but usually doesn't last.",
        "good_sign": "Moderate short interest + high days to cover + stock rallying = squeeze risk (exciting for longs).",
        "bad_sign": "Already-squeezed (volume spike, parabolic move) = squeeze likely over.",
        "emoji": "🚀",
    },
    "Performance vs S&P 500": {
        "full_name": "Relative Performance (Alpha)",
        "category": "Market Intelligence",
        "eli5": "Is the stock beating the S&P 500 or underperforming? If the S&P is up 10% and this stock is up 15%, it's outperforming (positive alpha).",
        "what_it_means": "Relative performance compares the stock to the S&P 500. Outperformance (positive alpha) suggests company-specific strength or market advantage. Underperformance suggests weakness or sector headwinds.",
        "good_sign": "Stock outperforming the S&P 500; alpha positive; outperformance consistent.",
        "bad_sign": "Stock significantly underperforming S&P 500; negative alpha; underperformance consistent.",
        "emoji": "📊",
    },
    "Sector Trend": {
        "full_name": "Sector ETF Momentum",
        "category": "Market Intelligence",
        "eli5": "Even great companies struggle if their whole industry is falling. Sector ETFs (like XLF for financials) show if the industry tailwinds are at your back or in your face.",
        "what_it_means": "Sector trends provide context. A stock beating the market might just ride sector momentum. A stock beating its sector is impressive. Compare the stock's performance to its sector ETF.",
        "good_sign": "Stock outperforming its sector ETF; sector ETF in an uptrend.",
        "bad_sign": "Stock underperforming its sector; sector ETF in a downtrend.",
        "emoji": "📈",
    },
    "VIX": {
        "full_name": "VIX (Volatility Index)",
        "category": "Market Intelligence",
        "eli5": "The market's 'fear gauge.' When VIX is high, everyone's nervous and selling. When VIX is low, everyone's calm and buying. Extreme moves often reverse.",
        "what_it_means": "The VIX measures implied volatility of S&P 500 options. VIX below 15 = calm market, above 30 = panic. Extreme VIX readings (very high or very low) often signal reversals or major moves ahead.",
        "good_sign": "VIX below 20 (calm, bullish); VIX spiking from low levels then receding (fear capitulation).",
        "bad_sign": "VIX above 30 (panic); VIX grinding higher over time (fear building).",
        "emoji": "😨",
    },
    "52-Week Rank": {
        "full_name": "52-Week High-Low Rank",
        "category": "Market Intelligence",
        "eli5": "Where is the stock trading in its 52-week range? At the high (100%) = strong momentum, at the low (0%) = beaten down. Rank of 75 = 75% of the way up from low to high.",
        "what_it_means": "52-week rank shows position within recent range. High rank (>75%) suggests strong momentum. Low rank (<25%) suggests it's been beaten down. Compare to price history to assess trends.",
        "good_sign": "52-week rank above 70% (in upper range); rank rising (new highs).",
        "bad_sign": "52-week rank below 30% (in lower range); rank falling (new lows).",
        "emoji": "📊",
    },

    # ========== VERDICTS & SCORES ==========
    "BUY": {
        "full_name": "BUY Verdict",
        "category": "Verdict",
        "eli5": "The analysis says: this stock looks undervalued and has good momentum. It could be a good time to buy. (But do your own research!)",
        "what_it_means": "A BUY verdict aggregates technicals, fundamentals, and sentiment. It suggests the stock is attractive at current levels with reasonable risk/reward. Doesn't guarantee returns.",
        "good_sign": "BUY verdict with high confidence; multiple BUY signals; BUY on pullbacks.",
        "bad_sign": "BUY verdict after the stock has already surged (buy high risk); BUY with low confidence.",
        "emoji": "🟢",
    },
    "HOLD": {
        "full_name": "HOLD Verdict",
        "category": "Verdict",
        "eli5": "The analysis says: this stock is okay, but not compelling to buy or sell right now. If you own it, hold. If you don't, wait for a better entry or pass.",
        "what_it_means": "A HOLD verdict suggests the stock is fairly valued or mixed signals. It could go either way. Good for patient holders, but lacks conviction for new buyers.",
        "good_sign": "HOLD after a big move (consolidating); HOLD on mixed technicals/fundamentals (balanced).",
        "bad_sign": "HOLD with declining fundamentals (deteriorating, might go to SELL).",
        "emoji": "🟡",
    },
    "SELL": {
        "full_name": "SELL Verdict",
        "category": "Verdict",
        "eli5": "The analysis says: this stock looks overvalued or is showing signs of weakness. It could be time to exit or avoid. (But consider tax implications if you own it.)",
        "what_it_means": "A SELL verdict suggests the stock is unattractive at current levels or showing deterioration. Risk/reward is unfavorable. Sell or avoid. Existing holders should consider exiting or trimming.",
        "good_sign": "SELL after stock surged (sell high); SELL with clear fundamental deterioration.",
        "bad_sign": "SELL after stock crashed (sell low in panic); SELL with low confidence.",
        "emoji": "🔴",
    },
    "Confidence Score": {
        "full_name": "Confidence Score (%)",
        "category": "Verdict",
        "eli5": "How sure is the analysis about the verdict? 95% = very sure, 60% = somewhat unsure. Higher confidence is better, but not a guarantee.",
        "what_it_means": "Confidence (0-100%) reflects signal alignment. High confidence (>80%) means technicals, fundamentals, and sentiment agree. Low confidence (<60%) means mixed signals.",
        "good_sign": "Confidence above 80% (strong agreement across signals).",
        "bad_sign": "Confidence below 60% (conflicting signals); confidence declining.",
        "emoji": "📊",
    },

    # ========== RISK FACTORS ==========
    "Earnings Proximity Risk": {
        "full_name": "Earnings Proximity Risk",
        "category": "Risk",
        "eli5": "Is an earnings announcement coming soon? If yes, the stock could be volatile. Before earnings, uncertainty is high. After earnings, volatility usually drops.",
        "what_it_means": "Earnings dates create uncertainty and volatility. Stocks can gap up or down on earnings surprise. Investors must consider earnings risk when setting stop losses and position sizes.",
        "good_sign": "Just-released earnings (uncertainty cleared); long time until next earnings (stability window).",
        "bad_sign": "Earnings coming this week (high volatility ahead); history of earnings misses (execution risk).",
        "emoji": "⚠️",
    },
}

CATEGORIES = ["Technical", "Fundamental", "Market Intelligence", "Risk", "Valuation", "Verdict"]


def get_term(term: str) -> dict:
    """
    Case-insensitive lookup of a term in the glossary.

    Args:
        term: The term to look up (e.g., "RSI", "rsi", "Relative Strength Index")

    Returns:
        dict with keys: full_name, category, eli5, what_it_means, good_sign, bad_sign, emoji
        None if term not found
    """
    if not term:
        return None

    # Direct lookup (case-insensitive)
    term_upper = term.upper()
    for key, data in GLOSSARY.items():
        if key.upper() == term_upper:
            return {**data, "term": key}

    # Fuzzy match on full_name
    term_lower = term.lower()
    for key, data in GLOSSARY.items():
        if term_lower in data["full_name"].lower() or data["full_name"].lower() in term_lower:
            return {**data, "term": key}

    return None


def get_by_category(category: str) -> list:
    """
    Get all terms in a specific category.

    Args:
        category: e.g., "Technical", "Fundamental", "Market Intelligence", etc.

    Returns:
        List of (term_name, term_data) tuples
    """
    if category not in CATEGORIES:
        return []

    results = []
    for term, data in GLOSSARY.items():
        if data["category"] == category:
            results.append((term, {**data, "term": term}))

    return sorted(results, key=lambda x: x[0])


def search_terms(query: str) -> list:
    """
    Search terms by name, full_name, or eli5 description.

    Args:
        query: Search string (case-insensitive)

    Returns:
        List of (term_name, term_data) tuples matching the query
    """
    if not query:
        return []

    query_lower = query.lower()
    results = []

    for term, data in GLOSSARY.items():
        # Check against term name, full_name, and eli5
        searchable = f"{term} {data['full_name']} {data['eli5']} {data['what_it_means']}".lower()

        if query_lower in searchable:
            results.append((term, {**data, "term": term}))

    return sorted(results, key=lambda x: x[0])


if __name__ == "__main__":
    # Quick test
    print("Example lookups:")
    print(f"RSI: {get_term('RSI')}")
    print(f"\nTechnical terms: {len(get_by_category('Technical'))} terms")
    print(f"Search for 'profit': {len(search_terms('profit'))} results")
