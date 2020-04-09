# Libre-Health-POC

## To deploy on local machine -

Use of a virtual environment is recommended.

1. Clone this repository
```
git clone https://github.com/aish-where-ya/Libre-Health-POC.git
```

2. Make sure you have tensorflow=1.15 installed. If not then run -
```
pip3 install tensorflow==1.15
```

3. Install requirements.txt
```
pip3 install -r requirements.txt
```

4. Run the project by navigating inside Libre-Health-POC and executing -
```
python3 index.py
```

5. Navigate to [http://127.0.0.1:5000/](http://127.0.0.1:5000/) and follow the given instructions.

## Note
The `requirements.txt` file installs `tensorflow==2.1.0`. This installation removes `tensorflow==1.15`. Please reinstall `tensorflow==1.15` using pip3 after running `pip3 install -r requirements.txt` so that both the versions of tensorflow are available for use.

## Images of the UI - 
1. Screen to enter details and upload X-Ray image.
![Screenshot from 2020-03-30 02-32-29](https://user-images.githubusercontent.com/32825331/78450970-1a852600-76a0-11ea-8436-834f2e882d0c.png)
2. Screen after prediction of the disease. (SSD Mobilenet shown in the picture)
![Screenshot from 2020-03-30 14-35-48](https://user-images.githubusercontent.com/32825331/78450973-21ac3400-76a0-11ea-9baa-e03dc5b4e06e.png)
