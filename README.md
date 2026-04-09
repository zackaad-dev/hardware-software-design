# Face Recognition Model for ESP32ES Sense 
1. Generate Model

```Bash
cd model
uv venv
uv run main.py
```
2. Generate Model Data CC file
```Bash
cd gen
xxd -i -C model.tflite > model_data.cc
sed -i "s/unsigned/extern const unsigned/g" model_data.cc
mv model_data.cc ../../esp32/main/model_data.cc
```

3. Update Model Data H file
```C
#pragma once

#include <stdint.h>

extern const unsigned char MODEL_TFLITE[];
extern const unsigned int MODEL_TFLITE_LEN;

#define IMG_SIZE // ADD IMAGE SIZE HERE - 224  
#define NUM_CLASSES // ADD NUMBER OF CLASSES HERE - 3
```

4. Build and Flash
Make sure that idf.py is in your path
```Bash
idf.py fullclean
idf.py build
idf.py flash
```






