install:
	pip install -r requirements.txt

run:
	flask --app app --debug run --port 3000
