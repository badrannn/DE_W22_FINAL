from airflow import DAG
from airflow.utils.dates import days_ago
from datetime import timedelta
from airflow.operators.python_operator import PythonOperator
from Milestone_1_template import ms1
import pandas as pd
from sqlalchemy import create_engine
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objs as go


def load_to_postgres(filename,filename2): 
    df = pd.read_csv(filename)
    df2=pd.read_csv(filename2)
    engine = create_engine("postgresql://root:root@postgres_accidents_datasets-pgdatabase-1:5432/UK_Accidents_2018")
    if(engine.connect()):
        print('connected succesfully')
    else:
        print('failed to connect')
    df.to_sql(name = 'UK_Accidents_2018',con = engine,if_exists='replace')
    df2.to_sql(name = 'lookup_table',con = engine,if_exists='replace')

def create_dashboard(filename):
    df = pd.read_csv(filename)

    x=df.number_of_casualties[df['urban_or_rural_area'] == 'Urban'].mean()
    y=df.number_of_casualties[df['urban_or_rural_area'] == 'Rural'].mean()

    fatal_counts = df[df['accident_severity'] == 'Fatal'].groupby("road_surface_conditions")['accident_severity'].count().astype(float)
    total_counts = df.groupby("road_surface_conditions")['accident_severity'].count().astype(float)
    fatal_percents = fatal_counts / total_counts * 100
    fig = px.bar(x=fatal_percents.index, y=fatal_percents.values, title="Percentage of Fatal Accidents by Road Surface Condition")

    road_type_counts = df[(df['day_of_week']=='Friday') & (df['light_conditions']!='Daylight')].groupby('road_type')['light_conditions'].count()
    data = [go.Pie(labels=road_type_counts.index, values=road_type_counts.values)]
    layout = go.Layout(title="Accidents by Road Type on Fridays with Non-Daylight Light Conditions")
    fig_pie = go.Figure(data=data, layout=layout)

    speed_limit_casualties = df.groupby(by="speed_limit")["number_of_casualties"].mean()
    data = [go.Scatter(x=speed_limit_casualties.index, y=speed_limit_casualties.values, mode='lines+markers')]
    layout = go.Layout(title="Average Number of Casualties by Speed Limit", xaxis_title="Speed Limit", yaxis_title="Number of Casualties")
    fig_line= go.Figure(data=data, layout=layout)

    app = dash.Dash()
    app.layout = html.Div(
    children=[
        html.H1(children="UK 2018 Accident dataset",),
        html.P(
            children="Urban vs Rural deaths average",
        ),
        dcc.Graph(
            figure={
                "data": [
                    {
                        "x": ["Urban", "Rural"],
                        "y": [x, y],
                        "type": "bar",
                    },
                ],
                "layout": {"title": "Urban vs Rural deaths average"},
            },
        ),
        html.H1(children="Accidents by Road Type on Fridays with Non-Daylight Light Conditions"),
        dcc.Graph(
            figure=fig_pie
        ),
        html.H1(children="Percentage of Fatal Accidents by Road Surface Condition"),
        dcc.Graph(
            figure=fig
        ),
        html.H1(children="Average Number of Casualties by Speed Limit"),
        dcc.Graph(
            figure=fig_line
        ),
        html.H1(children="Accidents per Weekday"),
        dcc.Graph(
            figure={
                "data": [
                    {
                        "x": df["day_of_week"].unique(),
                        "y": df["day_of_week"].value_counts(),
                        "type": "bar"
                    },
                ],
                "layout": {"title": "Accidents per Weekday"},
            }
        )
    ]
)
    app.run_server(host='0.0.0.0', port=8020, debug= False)
    print('dashboard is successful and running on port 8000')



default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    'start_date': days_ago(2),
    "retries": 0,
    "execution_timeout": timedelta(hours=2),
}

dag = DAG(
    'dag',
    default_args=default_args,
)

with DAG(
    dag_id = 'dag',
    schedule_interval = '@once',
    default_args = default_args,
    tags = ['clean_the_dataset'],
)as dag:
    milestone_1= PythonOperator(
        task_id = 'milestone_1',
        python_callable = ms1,
    )

    load_to_postgres_task=PythonOperator(
        task_id = 'load_to_postgres',
        python_callable = load_to_postgres,
        op_kwargs={
            "filename": "./data/output_dataset.csv",
            "filename2":"./data/lookup.csv"
        },
    )
    create_dashboard_task = PythonOperator(
        task_id = 'create_dashboard_task',
        python_callable = create_dashboard,
        op_kwargs={
            "filename": "./data/2018_Accidents_UK.csv"
        },
    )
    milestone_1>>load_to_postgres_task>>create_dashboard_task

    