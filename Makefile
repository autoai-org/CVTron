default:
	@echo "Usage:"
	@echo "\tmake test      # run pylint"

test:
	python tests/classifier.py
	python tests/detector.py