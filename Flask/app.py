from flask import Flask, render_template, request, redirect
from gensim.models.fasttext import FastText
import pickle
from bs4 import BeautifulSoup
import numpy as np
import os
import re
from nltk.stem import PorterStemmer
#debug command
# set FLASK_DEBUG=1

def docvecs(embeddings, docs):
    vecs = np.zeros((len(docs), embeddings.vector_size))
    for i, doc in enumerate(docs):
        valid_keys = [term for term in doc if term in embeddings.key_to_index]
        docvec = np.vstack([embeddings[term] for term in valid_keys])
        docvec = np.sum(docvec, axis=0)
        vecs[i,:] = docvec
    return vecs

app = Flask(__name__)
@app.route('/')
def home(): # homepage
    return render_template('home.html')

#routes for different categories
@app.route('/finance')
def finance():
    return render_template('finance.html')

@app.route('/healthcare')
def healthcare():
    return render_template('healthcare.html') #title = title, description = description, file= file_name)

@app.route('/sales')
def sales():
    return render_template('sales.html')

@app.route('/engineering')
def engineering():
    return render_template('engineering.html')

@app.route('/createjob', methods = ['GET', 'POST'])
def createjob():
    if request.method == 'POST':
        title = request.form['title']
        company = request.form['company']
        description = request.form['description']
        
        if request.form['button'] == 'classify job':
            tokenised_description = description.split(" ")

            descriptionFT = FastText.load('desc_FT.model')
            description_wv = descriptionFT.wv
            description_dv = docvecs(description_wv,[tokenised_description])

            with open('descFT_LR.pkl', 'rb') as f:
                model = pickle.load(f)
            
            y_hat = model.predict(description_dv)
            y_hat = y_hat[0]
            
            return render_template('createjob.html', prediction = y_hat, title = title, company = company, description = description)
    
        elif request.form['button'] == 'save job':
            category = request.form['category']
            if not category:
                message = 'Catergory cannot be empty'
                return render_template('createjob.html', title = title, company = company, description = description, category_flag = message)
            elif category not in ['Engineering', 'Healthcare_Nursing', 'Sales', 'Accounting_Finance']:
                message = 'Catergory must be on of the following: Engineering, Healthcare_Nursing, Sales or Accounting_Finance'
                return render_template('createjob.html', title = title, company = company, description = description, category_flag = message)
        
            else:
                soup = BeautifulSoup(open('templates/job_template.html'), 'html.parser')
                div_title = soup.find('div', {'class' : 'title'})
                soup_title = soup.new_tag('h1', id = 'title')
                soup_title.append(title)
                div_title.append(soup_title)

                div_company = soup.find('div', {'class' : 'company'})
                soup_company = soup.new_tag('h3',id = 'company')
                soup_company.append(company)
                div_company.append(soup_company)
                

                div_description = soup.find('div', {'class' : 'description'})
                soup_description = soup.new_tag('p')
                soup_description.append(description)
                div_description.append(soup_description)

                #getting the next int for new file 
                path = 'templates'
                file_names = []
                new_path = os.path.join(path,category)
                for file in os.listdir(new_path):
                    file_names.append(file)
                last_file = str(file_names[-1])
                x = re.findall("\d+",last_file)
                x = x[0].strip('0')
                filename = int(x) + 1
                filename = str(filename).zfill(5)
                filename = "/job" + filename + '.html' 
                with open('templates/' + category + filename, "w", encoding = 'utf-8') as f:
                    f.write(str(soup))
                return redirect(category + '/' + filename.replace('.html', ''))


            
    else:
        return render_template('createjob.html')
    


#route for viewing different jobs from different categories
@app.route('/<folder>/<filename>')
def job(folder, filename):
    return render_template('/' + folder + '/' + filename + '.html')

@app.route('/search', methods = ["POST"])
def search():
    if request.method == "POST":
        ps = PorterStemmer()
        search_string = request.form['searchword']
        search_string = ps.stem(search_string)

        article_search = []
        path = 'templates'
        for folder in os.listdir(path):
            if os.path.isdir(os.path.join(path,folder)):
                for filename in sorted(os.listdir(os.path.join(path,folder))):
                    if filename.endswith('html'):
                        with open(os.path.join(path,folder,filename), encoding = "utf8") as f:
                            description = f.read()
                        if search_string in description:
                            article_search.append([folder,filename.replace('.html',"")])
        results = len(article_search)
        return render_template("search.html",search_string = search_string, article_search = article_search)
    else:
        return render_template('home.html')