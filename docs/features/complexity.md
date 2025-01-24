# Algorithm complexity estimator

The following script is able to estimate the complexity of 
different approaches.

Steps:
## Generate instance properties
```bash
java -jar target/BMSSC-0.21-SNAPSHOT.jar --instance-properties
```

By default, the instance properties file is called `instance_properties.csv`.
See configuration TODO for a list of configurable parameters.

## Execute any experiment
Execute any of your experiments enabling metrics and the default 
solution JSON serializer. By enabling metrics, the framework will
keep track of how long it takes to execute each component,
and by enabling the default JSON serializer, the data is automatically
exported to the JSON solution files.

## Run the estimator
The current version of the estimator script is written in Python, and has
several dependencies that can be automatically installed with pip:
```bash
pip3 install -r src/main/resources/complexity/complexity-requirements.txt 
```

Both the script and the requirements file are included in any Mork project since version `0.21`. 
After the dependencies are installed, launch the script with:
```bash

```

See additional available options and examples using `--help`.