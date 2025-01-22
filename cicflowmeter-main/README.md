# Python CICFlowMeter

> This project is cloned from [Python Wrapper CICflowmeter](https://gitlab.com/hieulw/cicflowmeter) and customized to fit my need. Therefore, it is not maintained actively. If there are any problems, please create an issue or a pull request.  


### Installation
```sh
cd cicflowmeter
sudo make install
```

### Uninstall
```sh
sudo make uninstall
```

### Usage
```sh
usage: cicflowmeter [-h] (-i INPUT_INTERFACE | -f INPUT_FILE) [-c] [-u URL_MODEL] output

positional arguments:
  output                output file name (in flow mode) or directory (in sequence mode)

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_INTERFACE    capture online data from INPUT_INTERFACE
  -f INPUT_FILE         capture offline data from INPUT_FILE
  -c, --csv, --flow     output flows as csv
```


- Reference: https://www.unb.ca/cic/research/applications.html#CICFlowMeter
