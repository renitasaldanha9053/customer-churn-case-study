For running this project in your local,
You can download this project and maybe open in an editor such as VSCode.
Once you have opened it in an editor, say VSCode. Select View -> Terminal.
Click on New Terminal and select cmd as the terminal.
You need to first install all the additional python packages in order to run this project.
The additional packages are mentioned in the requirements.txt. Having Python installed 
on your local machine is a prerequisite.
Then you need to just open a new terminal as instructed above and can run the below command:

pip install requirements.txt

You can run the notebooks one by one in order and understand what's happening in the code.
Notebook 1 is Exploratory Data Analysis
Notebook 2 is Model Prototype  and 
Notebook 3 is Model Evaluation

In case you want to run the dashboards:

Navigate to src/dashboards from the terminal 

cd dashboards

Then run the following:

streamlit run [dashboard_name].py


