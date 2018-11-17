from flask import render_template, request,flash, redirect, url_for
from datetime import datetime
from app.extract_and_clean_tweets import extract_tweets_lazy, get_only_chars
import re
from app import app

@app.context_processor
def inject_now():
    return {'now': datetime.utcnow()}

@app.route('/', methods =['GET', 'POST'])
def index():
    if request.method == 'POST':
        message = request.form['twitter_handle']
        if message:
            raw_data = extract_tweets_lazy(message)
            if raw_data and len(raw_data) > 1 :
                clean_data = get_only_chars(raw_data)
                predicted_personality = 'ejts'
                #predicted_personality = predict_personality(clean_data)
                confidence_level = '90'


                flash(u'Tweets extracted successfully', 'danger')
                return render_template('index.html', clean_data = clean_data, raw_data= raw_data, 
                predicted_personality = predicted_personality, confidence_level= confidence_level)
            else:
                flash(u'Provide a valid twitter handle', 'danger')
                return redirect(url_for('index'))
        else:
            flash(u'Provide a valid twitter handle', 'danger')
            return redirect(url_for('index'))
    if request.method == 'GET':
        return render_template('index.html')


@app.route('/about')
def about():
    return render_template("about.html")
    