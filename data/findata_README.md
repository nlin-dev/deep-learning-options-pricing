# Options Contracts Fundamental Greeks for Calls and Puts

## Abstract

The original financial information this dataset was based off contained historical options trading data for Tesla (TSLA), Apple (AAPL), and Google (GOOG). Some of the fundamental Greeks it encompasses are contract strike price, bid size, ask size, volume, implied volatility, delta, and theta. This dataset was prepared with the intent to be trained in various Recurrent Neural Networks as options pricing is a complex task that requires analyzing various market factors and dynamics. Traditional models, which are trained on much more simplistic stock data, often struggles to capture the complex relationships within options data.

The datasets which comprise this dataset can be found here: https://drive.google.com/drive/folders/14UDaBowufW9BTIz6CvdKxtkL2l0c_DAB?usp=sharing 

These datasets contain both call and put data and targets, as well as a general options pricing dataset for Tesla (TSLA).


## Data Collection

The original financial information was collected through publicly accessible trading platforms through various APIs and then stored centrally on IVolatility. However access to this financial information is no longer accessible, as some weeks after we first accessed the original data, the financial information was removed from the site. From here we worked on manipulating the data so that various deep learning models, such as our intended RNNs, could be trained on the data. Examples of this data manipulation would be implementation of our own Greeks, additional features, and creating our own target variable.


## Data Description

The original financial information dataset was structured as follows: 
- **strike**: the price at which the holder of the option can buy or sell the underlying asset, in terms of calls and puts respectively
- **size_bid**: number of contracts available at the bid price, reflects demand for the option
- **size_ask**: number of contracts available at the ask price, reflects supply for the option
- **volume**: total number of options contacts that have been traded during a specific time period
- **implied_volatility**: reflects market's view on likelihood of changes in underlying asset's price which affects the options premium
- **price_opt**: current market price of the option
- **delta**: measures rate of change in options price in relation to one unit change in price of underlying asset, or options price change in comparison to stock actual value change
- **gamma**: represents rate of change in delta for an one-unit change in asset price
- **theta**: measures rate of change in options price with respect to time, reflecting loss in value as option approaches expiration date
- **vega**: indicates option's price changes to 1% change in implied volatility, showing sensitivity to implied volatility
- **rho**: measures sensitivity of options price to a 1% change in interest rates, indicates how interest rate changes options value
- **pre_iv**: implied volatility of option from previous trading day

Additional financial information was added to the dataset in order to make it easier to train deep learning models on. Examples of such would be scaling Greeks such as delta and theta by different factors in order to have a better standardize relationship with other variables. This can be seen in the dataset.

Finally in order for the deep learning model to be trained, we had to determine the most optimal way to set a target variable based on the original historical options trading data. We were able to achieve this by implementing a simple price variable to determine optimal buy point. We accomplished this by simply finding the difference between bid price and the ask price for each individual option.


## Methodology

This data was compiled with the intention of providing a robust dataset and foundation for the development of various deep learning models. By taking historical options trading data, we were able to work with financial information which was both accurate and relevant to train our models on. However we quickly determined that the Greek fundamentals of the options were not standardized enough to remove the noise that would result from training the models we intended to use. Because of this we ran through several cycles of hyperparameter optimization as well as imputation of additional variables to make the dataset more robust and faster for model compilation. 


## Usage

This dataset can be used in various ways to train deep learning models in order to predict optimal options prices. We implemented this curated dataset in order to train four different deep learning models:
1. Multilayer Perceptron
2. Recurrent Neural Netowrk (RNN)
3. Long Short-Term Memory (LSTM)
4. Gated Recurrent Unit (GRU)

However other uses of this dataset could potentially be: 
1. Time Series Analysis for Price Movement Prediction
2. Volume Analysis for Market Sentiment Indication
3. Option-type Analysis for Business Strategy Development

## License 

The original financial information this dataset was based off of was a dataset made available through . After modification this dataset is now openly available under the Open Data Commons Open Database Licesne (ODbL). Given this, you are free to use, modify, and share this dataset provided you attribute the source as well as share any derivative works under the same license.


## Citing the Dataset

Please cite this dataset as follows:

Lin, Nicholas. Zhou, Roy. (2023). Options Contracts Fundamental Greeks for Calls and Puts. NYU Center for Data Science. 


## Contact Information

For questions or contributions to this dataset, please contact:

Nicholas Lin
Email: nl2872@columbia.edu
ORCID ID: 0009-0003-9669-8182

Roy Zhou
Email: rz1478@nyu.edu
Data Scientist, BNY Mellon
