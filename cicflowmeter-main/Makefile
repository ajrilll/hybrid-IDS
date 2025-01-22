VERSION:=$(shell python setup.py --version)

install:
	python3 setup.py install

uninstall:
	pip uninstall cicflowmeter -y

clean:
	sudo rm -r /home/ajril/Skripsi/Tools/cicflowmeter-main/build
	sudo rm -r /home/ajril/Skripsi/Tools/cicflowmeter-main/dist
	sudo rm /usr/local/bin/cicflowmeter

build:
	python3 setup.py sdist bdist_wheel --universal

release:
	@git tag -a v$(VERSION)
	@git push --tag
