# ML & Data Analysis Notebooks

A collection of my machine-learning and data-analysis notebooks from 2024–2025, gathered in one place. Each folder used to be its own repository; its commit history is preserved here.

These are learning projects and experiments, not production systems. The notes below say plainly what worked and what didn't. For production-style work, see my [other repositories](https://github.com/dinulimann).

| Folder | What it is | Main tools |
|---|---|---|
| [crypto-chatbot-forecast](crypto-chatbot-forecast/) | Telegram chatbot for crypto questions with an LSTM intent classifier, voice replies and an LSTM price forecast (undergraduate thesis project) | TensorFlow/Keras, NLTK, yfinance, python-telegram-bot, gTTS |
| [lstm-chatbot-forecast](lstm-chatbot-forecast/) | Second iteration: Bi-LSTM/GRU intent model and a BTC price forecast | TensorFlow/Keras, yfinance |
| [sentiment-telegram-bot](sentiment-telegram-bot/) | 3-class text sentiment model served through a Telegram bot | Keras, python-telegram-bot |
| [seq2seq-chatbot](seq2seq-chatbot/) | Encoder–decoder (seq2seq) LSTM chatbot experiment | TensorFlow/Keras, NLTK |
| [car-price-analysis](car-price-analysis/) | Exploratory analysis of a car price dataset | pandas, seaborn |
| [ad-campaign-analysis](ad-campaign-analysis/) | Exploratory analysis of advertising campaign performance | pandas, seaborn |
| [elden-ring-analysis](elden-ring-analysis/) | Elden Ring weapon stats analysis, plus a small Q&A text dataset and model sketch | pandas, seaborn, Keras |

## Notes per project

### crypto-chatbot-forecast
- `model chatbot/chattobottoo.ipynb`: a Telegram bot. An Embedding + LSTM classifier maps the user's message to an intent; the bot replies with text and a gTTS voice message, and fetches live BTC/ETH/BNB prices with yfinance.
- `model forecast/prediksieth.ipynb`: a 2-layer LSTM that predicts the next daily close from the previous 60 days (80/20 time split, MinMax scaling).
- `app.ipynb`: the combined app with speech recognition input.
- **Known issues:** the "RMSE" cell computes `sqrt(mean(error)**2)`, which is the absolute mean error (bias), not RMSE. The correct form is `sqrt(mean(error**2))`. The cell that loads the price data into `df` is missing from the saved notebook, so it does not run top to bottom. The notebooks were written for Google Colab and read model files from Google Drive.

### lstm-chatbot-forecast
- `ModelChatbot.ipynb`: intent classifier experiments with Bidirectional LSTM and GRU layers.
- `ModelForecast.ipynb`: LSTM forecast of BTC-USD daily close.
- `Chatbot.ipynb` is the same code as `crypto-chatbot-forecast/model chatbot/chattobottoo.ipynb`.
- **Known issues:** the same RMSE formula issue as above. The model is compiled with an `accuracy` metric, which is meaningless for regression and always shows 0.

### sentiment-telegram-bot
- `modelling.ipynb`: tokenizer (20,000 words, sequences of 35 tokens) + Embedding + Dense softmax over 3 sentiment classes, trained for 10 epochs.
- `telegram.ipynb`: loads the saved `model.h5` and `tokenizer.pickle` and answers each Telegram message with its predicted sentiment. Set your own bot token in `TOKEN`.
- **Result:** 93% training accuracy but **70% validation accuracy**, so the model overfits. Dropout, early stopping or a pretrained embedding would be the next steps.

### seq2seq-chatbot
- Encoder–decoder LSTM on an intents dataset from Kaggle.
- **Result:** training logs show loss and accuracy stuck at exactly 0, which means the training setup (targets or loss) is broken. It is kept as an experiment, not a working model.

### car-price-analysis, ad-campaign-analysis, elden-ring-analysis
- Exploratory data analysis with pandas and seaborn: distributions, correlations, top-N rankings and written conclusions (car price notebook, in Indonesian).
- Data is downloaded from Kaggle with `kagglehub` (a copy of each CSV is included).
- `elden-ring-analysis/modelling.ipynb` is only an unfinished model sketch.

## Running

Most notebooks were written for Google Colab. Open one from GitHub in Colab, or run it locally with Jupyter after installing what its first cells import. Kaggle downloads need Kaggle credentials for `kagglehub`.
