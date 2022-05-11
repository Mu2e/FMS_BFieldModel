import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash.dependencies import Input, Output
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_table
from dash_table.Format import Format, Scheme

# SolCalc
from helicalc import helicalc_dir, helicalc_data
from helicalc.solcalc import SolCalcIntegrator
from helicalc.geometry import read_solenoid_geom_combined
from helicalc.cylinders import get_thick_cylinders_padded

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# load nominal PS geom
# paramdir = '/home/ckampa/coding/helicalc/dev/params/'
paramdir = helicalc_dir + 'dev/params/'
paramfile = 'Mu2e_V13'

df_PS_nom = read_solenoid_geom_combined(paramdir, paramfile).iloc[:3]

# calculate layer thickness
# FIXME!

# integration params
drz = np.array([5e-3, 1e-2])

# editable vs. dependent columns
cols_edit = ['Ri', 'x', 'y', 'z', 'rot0', 'rot1', 'rot2', 'N_layers',
             'N_turns', 'I_turn']
cols_stat = ['Coil_Num', 'Ro', 'L', 'I_tot', 'N_turns_tot', 'helicity', 'h_cable',
             'w_cable', 'h_sc', 'w_sc', 't_gi', 't_ci', 't_il', 'phi0_deg', 'phi1_deg',
             'pitch']


# load TS+DS contribution to PS
#PSoff_file = '/home/shared_data/Bmaps/SolCalc_complete/Mau13.SolCalc.PS_region.standard.PSoff.pkl'
PSoff_file = helicalc_data+'Bmaps/aux/Mau13.SolCalc.PS_region.standard.PSoff.pkl'
df_PSoff = pd.read_pickle(PSoff_file)
df_PSoff = df_PSoff.astype(float)
# m = (df_PSoff.Y == 0.) & (np.isin(df_PSoff.X - 3.904, [0., 0.4, 0.7]))
m = (df_PSoff.Y == 0.) & (np.isin(df_PSoff.X, [3.904, 4.304, 4.604]))
df_PSoff_lines = df_PSoff[m].copy().reset_index(drop=True, inplace=False)
# print(df_PSoff_lines)

# formatting/style
green = 'rgb(159, 210, 128)'
plot_bg = 'rgb(240, 240, 240)'

button_style = {'fontSize': 'large',
                'backgroundColor': green,
                }

# plot globals
marker_size = 10
fsize_plot = 20
fsize_ticks = 14

# instantiate app
app = dash.Dash(name='solcalc', external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1('SolCalc Magnet Builder (Production Solenoid)'),
    # html.H2('Coils Plot'),
    dcc.Graph(id='coils-plot'),
    html.H2('Coil Geometries'),
    # tables
    html.H3('Editable Parameters'),
    dash_table.DataTable(id='editable-table',
                         columns=[{'name':i, 'id': i, 'hideable':True, 'type':'numeric',
                         'format': Format(scheme=Scheme.fixed, precision=4),} for i in cols_edit],
                         data=df_PS_nom[cols_edit].to_dict('records'),
                         editable=True),
    html.Br(),
    html.Button('Recalculate Field', id='calc-button', style=button_style),
    # field plot
    html.H2('Field Plot'),
    html.Label('Plotting Options:'),
    html.Label('Field Component:'),
    dcc.Dropdown(
        id='yaxis-column-field',
        options=['Bx', 'By', 'Bz'],
        value='Bz',
        multi=False,
        #style=desc_style,
    ),
    html.Label('Field value or gradient?'),
    dcc.RadioItems(
        id='yaxis-type-field',
        options=[{'label': i, 'value': i} for i in ['B_i', 'grad_z(B_i)']],
        value='B_i',
        labelStyle={'display': 'inline-block'},
        #style=desc_style,
    ),
    html.Label('Include TS/DS Contribution?'),
    dcc.RadioItems(
        id='include-TS-field',
        options=[{'label': i, 'value': i} for i in ['yes', 'no']],
        value='yes',
        labelStyle={'display': 'inline-block'},
        #style=desc_style,
    ),
    html.Label('Individual coil contributions or combined field?'),
    dcc.RadioItems(
        id='indiv-contrib',
        options=[{'label': i, 'value': i} for i in ['combined', 'individal']],
        value='combined',
        labelStyle={'display': 'inline-block'},
        #style=desc_style,
    ),
    html.Label('Field unit:'),
    dcc.RadioItems(
        id='field-unit',
        options=[{'label': i, 'value': i} for i in ['Gauss', 'Tesla']],
        value='Gauss',
        labelStyle={'display': 'inline-block'},
        #style=desc_style,
    ),
    dcc.Graph(id='field-plot'),
    # FIXME!
    # not positive best placement for these
    html.H3('Static/Dependent Parameters'),
    dash_table.DataTable(id='static-table',
                         columns=[{'name':i, 'id': i, 'hideable':True, 'type':'numeric',
                         'format': Format(scheme=Scheme.fixed, precision=4),} for i in cols_stat],
                         data=df_PS_nom[cols_stat].to_dict('records'),
                         editable=False),
    html.H3('Notes on Dependent Parameters'),
    # dcc.Markdown('''
    # $R_o = R_i + h_{cable}*N_{layers} + 2*t_{gi} + 2*t_{ci}*N_{layers} + 2*{t_il}*(N_{layers}-1)$
    # '''),
    #html.Div(html.P(['Notes on depdendent parameters:', html.Br(),
    html.Div(html.P([
                    'Ro = Ri + h_cable*N_layers + 2*t_gi + 2*t_ci*N_layers + 2*t_il*(N_layers-1)', html.Br(),
                    'pitch = h_cable + 2*t_ci', html.Br(),
                    'L = pitch*N_turns + 2*t_gi [note nominal seems to use (N_turns-1)]', html.Br(),
                    'N_turns_tot = N_turns * N_layers', html.Br(),
                    'I_tot = I_turn * N_turns_tot',])),
    # hidden divs for data
    html.Div(children=df_PS_nom[cols_edit+cols_stat].to_json(),
             id='geom-data', style={'display': 'none'}),
    html.Div(id='field-data', style={'display': 'none'}),
])

# update geom div when button is clicked
@app.callback(
    [Output('geom-data', 'children'),
     Output('static-table', 'data'),],
    [Input('calc-button', 'n_clicks'),],
    [State('static-table', 'data'),
     State('static-table', 'columns'),
     State('editable-table', 'data'),
     State('editable-table', 'columns')],
)
def update_geom_data(n_clicks, rows_stat, cols_stat, rows_edit, cols_edit):
    # load data
    df_edit = pd.DataFrame(rows_edit, columns=[c['name'] for c in cols_edit], dtype=float)
    print(df_edit)
    print(df_edit.info())
    df_stat = pd.DataFrame(rows_stat, columns=[c['name'] for c in cols_stat], dtype=float)
    # calculations
    df_stat.loc[:, 'Ro'] = df_edit.Ri + df_stat.h_cable * df_edit.N_layers + \
    2 * df_stat.t_gi + 2*df_stat.t_ci*df_edit.N_layers +\
    2*df_stat.t_il*(df_edit.N_layers - 1)
    df_stat.loc[:, 'L'] = df_stat.pitch * df_edit.N_turns + 2 * df_stat.t_gi
    df_stat.loc[:, 'N_turns_tot'] = df_edit.N_turns * df_edit.N_layers
    df_stat.loc[:, 'I_tot'] = df_edit.I_turn + df_stat.N_turns_tot
    # combine results
    df = pd.concat([df_stat, df_edit], axis=1)
    return df.to_json(), df_stat.to_dict('records')

# update coils plot
@app.callback(
    Output('coils-plot', 'figure'),
    [Input('geom-data', 'children'),],
)
def plot_coils(df):
    df = pd.read_json(df)
    # get cylinders PS
    xs, ys, zs, cs = get_thick_cylinders_padded(df, [1, 2, 3])
    # get cylinders nominal PS
    xs_n, ys_n, zs_n, cs_n = get_thick_cylinders_padded(df_PS_nom, [1, 2, 3])
    # FIXME! Add some of the TS coils
    # return surface plot
    # layout
    # camera
    # y up
    # camera = dict(
    #     up=dict(x=0, y=1, z=0),
    #     #center=dict(x=-3.904, y=0, z=9.),
    #     eye=dict(x=-2, y=0., z=0.)
    # )
    # z up
    camera = dict(
        up=dict(x=0, y=0, z=1),
        #center=dict(x=-3.904, y=0, z=9.),
        eye=dict(x=0., y=-2., z=0.)
    )
    layout = go.Layout(
        title='Coil Layout',
        height=700,
        font=dict(family="Courier New", size=fsize_plot,),
        margin={'l': 60, 'b': 60, 't': 60, 'r': 60},
        scene=dict(aspectmode='data', camera=camera,
                   xaxis={'title': 'Z [m]', 'tickfont':{'size': fsize_ticks}},
                   yaxis={'title': 'X [m]', 'tickfont':{'size': fsize_ticks}},
                   zaxis={'title': 'Y [m]', 'tickfont':{'size': fsize_ticks}},),
        plot_bgcolor=plot_bg,
        # autosize=True,
        # width=1600,
        # height=800,
    )

    return {'data':
    #[go.Surface(x=xs, y=ys, z=zs, surfacecolor=cs,
    [go.Surface(x=zs_n, y=xs_n, z=ys_n, surfacecolor=cs_n,
                 colorscale=[[0,'rgba(0,0,0,0)'],[1,'rgba(220, 50, 103, 0.8)']],
                 showscale=False,
                 showlegend=True,
                 opacity=1.0,
                 name='PS Coils (nominal)',),
    go.Surface(x=zs, y=xs, z=ys, surfacecolor=cs,
                 colorscale=[[0,'rgba(0,0,0,0)'],[1,'rgba(138, 207, 103, 0.8)']],
                 showscale=False,
                 showlegend=True,
                 opacity=1.0,
                 name='PS Coils (current)',),
    ],
    'layout': layout,
    }

# recalculate field
@app.callback(
    Output('field-data', 'children'),
    [Input('geom-data', 'children'),],
)
def calculate_field(df):
    df = pd.read_json(df)
    # create dataframe with same grid as PSoff filtered dataframe
    df_calc = df_PSoff_lines[['X', 'Y', 'Z', 'R']].copy()
    for i in range(len(df)):
    # for geom in geom_df_mu2e.itertuples():
        j = int(round(df.iloc[i].Coil_Num))
        # print coil number to screen for reference
        #print(f'Calculating coil {i+1}/'+f'{N_coils}', file=old_stdout)
        # instantiate integrator
        mySolCalc = SolCalcIntegrator(df.iloc[i], drz=drz)
        # integrate on grid (and update the grid df)
        df_calc = mySolCalc.integrate_grid(df_calc, N_proc=1)
        # save single coil results
        # mySolCalc.save_grid_calc(savetype='pkl',
        #                          savename=datadir+base_name+f'.coil_{j}',
        #                          all_solcalc_cols=False)
    # combine fields
    for i in ['x', 'y', 'z']:
        cols = []
        for col in df_calc.columns:
            if f'B{i}_solcalc' in col:
                # T to G
                df_calc.eval(f'{col} = {col} * 1e4', inplace=True)
                cols.append(col)
        eval_str = f'B{i} = '+'+'.join(cols)
        df_calc.eval(eval_str, inplace=True, engine='python')
    print(df_calc)
    print(df_calc.info())
    return df_calc.to_json()

# update plot
@app.callback(
    Output('field-plot', 'figure'),
    [Input('field-data', 'children'),
     Input('yaxis-column-field', 'value'),
     Input('yaxis-type-field', 'value'),
     Input('include-TS-field', 'value'),
     Input('indiv-contrib', 'value'),
     Input('field-unit', 'value')],
)
def field_plot(df, ycol, ytype, incTS, plotIndiv, unit):
    # save original unit
    unit_ = unit
    unit_print = unit
    #print(df)
    df = pd.read_json(df)
    #print(df)
    xs = df.X.unique()
    #print(xs)
    rs = xs - 3.904
    # shared calculations
    zs = df.Z.values
    m1 = (df.X == xs[0])
    m2 = (df.X == xs[1])
    m3 = (df.X == xs[2])
    ms = [m1, m2, m3]
    # plotting depends most heavily on whether plotting individual coils
    if plotIndiv == 'combined':
        B = df[ycol].values.astype(float)
        if incTS == 'yes':
            B += df_PSoff_lines[ycol].values
        if unit == 'Tesla':
            B *= 1e-4
        t_inc = ''
        if ytype == 'grad_z(B_i)':
            ycol = f'grad_z({ycol})'
            unit_print = unit_+'/m'
            t_inc = ' Gradient'
            for m_ in ms:
                B[m_] = np.concatenate([[np.nan],np.diff(B[m_]) / np.diff(zs[m_])])
        data = [go.Scatter(x=zs[m_], y=B[m_], mode='lines+markers',
                           marker={'color':c, 'size': marker_size, 'opacity': 0.85,
                           'line': {'width':0.1, 'color': 'white'}},
                           line={'width':1, 'color': c},
                           name=f'R = {r:0.2f}'
                           ) for m_, c, r in zip(ms, ['blue', 'green', 'red'], rs)]
    else:
        cs_list = [['blue', 'green', 'red'],
                   ['purple', 'lime', 'pink'],
                   ['cyan', 'darkgreen', 'orange']]
        data = []
        ycols_PS = [1, 2, 3]
        ycols_TS = ['']
        for yc, cs in zip(ycols_PS, cs_list):
            yc_full = ycol+f'_solcalc_{yc}'
            B = df[yc_full].values.astype(float)
            if unit == 'Tesla':
                B *= 1e-4
            t_inc = ''
            if ytype == 'grad_z(B_i)':
                ycol_ = f'grad_z({ycol})'
                unit_print = unit_+'/m'
                t_inc = ' Gradient'
                for m_ in ms:
                    B[m_] = np.concatenate([[np.nan],np.diff(B[m_]) / np.diff(zs[m_])])
            else:
                ycol_ = ycol
            for m_, c, r in zip(ms, cs, rs):
                data.append(go.Scatter(
                    x=zs[m_], y=B[m_], mode='lines+markers',
                    marker={'color':c, 'size': marker_size, 'opacity': 0.85,
                    'line': {'width':0.1, 'color': 'white'}},
                    line={'width':1, 'color': c},
                    name=f'R = {r:0.2f}, Coil {yc}'))
        # make another trace for TS if necessary
        if incTS == 'yes':
            B = df_PSoff_lines[ycol].values.astype(float)
            if unit == 'Tesla':
                B *= 1e-4
            t_inc = ''
            if ytype == 'grad_z(B_i)':
                ycol_ = f'grad_z({ycol})'
                unit_print = unit_+'/m'
                t_inc = ' Gradient'
                for m_ in ms:
                    B[m_] = np.concatenate([[np.nan],np.diff(B[m_]) / np.diff(zs[m_])])
            else:
                ycol_ = ycol
            for m_, c, r in zip(ms, ['black', 'brown', 'yellow'], rs):
                data.append(go.Scatter(
                    x=zs[m_], y=B[m_], mode='lines+markers',
                    marker={'color':c, 'size': marker_size, 'opacity': 0.85,
                    'line': {'width':0.1, 'color': 'white'}},
                    line={'width':1, 'color': c},
                    name=f'R = {r:0.2f}, TS+DS Coils'))

    # layout should work with all configurations
    layout = go.Layout(
        title=f'Field{t_inc} Plot: y==0.0 m',
        height=700,
        font=dict(family="Courier New", size=fsize_plot,),
        margin={'l': 60, 'b': 60, 't': 60, 'r': 60},
        scene=dict(aspectmode='auto',
                   #xaxis={'title': 'Z [m]', 'tickfont':{'size': fsize_ticks}},
                   #yaxis={'title': f'{ycol} [{unit}]', 'tickfont':{'size': fsize_ticks}},
        ),
        xaxis={'title': 'Z [m]', 'tickfont':{'size': fsize_ticks}},
        yaxis={'title': f'{ycol} [{unit_print}]', 'tickfont':{'size': fsize_ticks}},
        plot_bgcolor=plot_bg,
        showlegend=True,
    )

    return {'data':data, 'layout':layout}


if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1')
