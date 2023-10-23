# IntentClassification


Install the required dependencies into the virtual environment

```pip install -r requirements.txt```

Run the uvicorn standard command to run the app.py --- to run the model with lstm classifier

```uvicorn app:app --host 0.0.0.0 --port 8000 --reload```  

Run the uvicorn standard command to run the appRoberta.py --- to run the model with lstm classifier

```uvicorn app:appRoberta --host 0.0.0.0 --port 8000 --reload``` 

Use Postman POST api to check the intent of the statment -  use the below url for post action.
```http://0.0.0.0:8000/intent```

Also can give the query using the browser

```http://0.0.0.0:8000/```

# Training the model

```cd training```

```python3 training_code.py```
