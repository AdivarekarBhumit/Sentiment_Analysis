import pickle as pb
import logging
import tensorflow as tf 

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras import backend as k
from telegram.ext import CommandHandler, Filters, MessageHandler, Updater, ConversationHandler

import utils

# Deep learning model functions
with open('../Flask app/helpers/helper.pb', 'rb') as f:
    args = pb.load(f)
vocab_to_int = args['vocab_to_int']
int_to_vocab = args['int_to_vocab']
maxlen = args['maxlen']

def get_model():
    return load_model('sentiment.hdf5')

def predict_sentiment(sequences):
    #k.clear_session()
    model = get_model()
    preds = model.predict(sequences)
    return preds[0]

def text_to_seq(text, max_len):
    text = utils.clean_text(text)
    seq = []
    for word in text.split():
        seq.append(vocab_to_int[word])
    pad_seq = pad_sequences([seq], maxlen=max_len)
    return pad_seq

# Telegram API Functions
# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def error(bot, update, error):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, error)


def start(bot, update):
    bot.send_message(update.message.chat_id, text='Enter your message to check its sentiment.')

def get_text(bot, update):
    msg = update.message.text
    prediction = predict_sentiment(text_to_seq(msg, maxlen))
    print(prediction)
    bot.send_message(update.message.chat_id, text='Your text is {:.2f}% positive and {:.2f}% negative'.format(prediction[1] * 100.00, prediction[0] * 100.00))

def main():
    TOKEN = open('P:/SentriBot API key/sentri.txt').read()

    updater = Updater(TOKEN)

    dp = updater.dispatcher

    # Adding Command Handler
    cmd_helper = CommandHandler('start', start)
    cmd_helper2 = CommandHandler('again', start)
    # dp.add_handler(MessageHandler(Filters.text | Filters.photo | Filters.location, echo))
    dp.add_handler(cmd_helper)
    dp.add_handler(cmd_helper2)
    dp.add_handler(MessageHandler(Filters.text, get_text))

    # print(content)

    # log all errors
    dp.add_error_handler(error)

    updater.start_polling()

    updater.idle()

if __name__ == '__main__':
    main()