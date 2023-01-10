import datetime
import os

# Get model ID
model_id = input("Model ID: ")
# Get data source M for mohanta, R for rajagopalan
while True:
    try:
        data_source = input("Data source (M/R): ")
        if data_source == "M":
            data_source = "mohanta"
            break
        elif data_source == "R":
            data_source = "rajagopalan"
            break
        else:
            print("Invalid data source. Please enter M or R.")
            raise ValueError
    except ValueError:
        continue

# ensure all capital letters
model_id = model_id.upper()
# open optimization_layout.sh file and replace MODELNAME with model_id
with open("bash\\optimization_layout.sh", "r") as f:
    lines = f.readlines()

## create folder for model
# get date as MMDDYY
date = datetime.datetime.now().strftime("%m%d%y")
# check if folder exists
if not os.path.exists(f"Z:\\FlYMazeRL_ChoiceEngg\\Optimal_Schedules\\acceptreject\\mohanta2022\\{model_id}"):
    os.mkdir(f"Z:\\FlYMazeRL_ChoiceEngg\\Optimal_Schedules\\acceptreject\\mohanta2022\\{model_id}")
    os.mkdir(f"Z:\\FlYMazeRL_ChoiceEngg\\Optimal_Schedules\\acceptreject\\mohanta2022\\{model_id}\\{date}")
else:
    i = 1
    if not os.path.exists(
        f"Z:\\FlYMazeRL_ChoiceEngg\\Optimal_Schedules\\acceptreject\\mohanta2022\\{model_id}\\{date}"
    ):
        os.mkdir(f"Z:\\FlYMazeRL_ChoiceEngg\\Optimal_Schedules\\acceptreject\\mohanta2022\\{model_id}\\{date}")
    else:
        while os.path.exists(
            f"Z:\\FlYMazeRL_ChoiceEngg\\Optimal_Schedules\\acceptreject\\mohanta2022\\{model_id}\\{date}_{i}"
        ):
            i += 1
        os.mkdir(f"Z:\\FlYMazeRL_ChoiceEngg\\Optimal_Schedules\\acceptreject\\mohanta2022\\{model_id}\\{date}_{i}")
        date = f"{date}_{i}"

## create optimizer script
# remove old optimization_{model_id}.sh file
if os.path.exists(f"optimization_{model_id}.sh"):
    os.remove(f"optimization_{model_id}.sh")
with open(f"optimization_{model_id}.sh", "w") as f:
    # replace MODELNAME with model_id and DATE with date
    for line in lines:
        if "MODELNAME" in line:
            line = line.replace("MODELNAME", model_id)
        if "DATE" in line:
            line = line.replace("DATE", date)
        if "DATASOURCE" in line:
            line = line.replace("DATASOURCE", data_source)
        f.write(line)

## use chmod to make optimizer script executable
os.system(f"chmod +x optimization_{model_id}.sh")

## run optimizer script and wait for it to finish
# os.system(f"./optimization_{model_id}.sh")

## print a reminder to delete the optimizer script after it finishes
print(f"Remember to delete optimization_{model_id}.sh after it finishes!")

