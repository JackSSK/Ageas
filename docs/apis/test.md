### [Tutorial](https://nkmtmsys.github.io/Ageas/tutorial)


# Test()
Function to test whether AGEAS is performing normally or not using sample GEMs in ageas/test/.
> ageas.Test(cpu_mode:bool = False)


## **Args**

+ **_cpu_mode_**: Default = False

    If cpu_mode is on, AGEAS will be forced to only use CPU. Otherwise, AGEAS will automatically select device, preferring GPUs.


## **Outputs**
+ ageas._main.Launch object


## **Example**
```python
import ageas
result = ageas.Test(cpu_mode = True)
```
