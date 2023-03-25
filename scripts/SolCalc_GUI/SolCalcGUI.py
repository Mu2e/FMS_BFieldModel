import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash.dependencies import Input, Output
from dash import dcc
from dash import html
from dash import ctx
from dash.dependencies import Input, Output, State
import dash_table
from dash_table.Format import Format, Scheme

# SolCalc
from helicalc import helicalc_dir, helicalc_data
from helicalc.solcalc import SolCalcIntegrator
from helicalc.geometry import read_solenoid_geom_combined
from helicalc.cylinders import get_thick_cylinders_padded
# additional code & info for Mu2e-II PS
sys.path.append(helicalc_dir+'scripts/SolCalc_GUI/')
from requirements_Mu2e import *
from resistive_power import *

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# globals
resistivities_dict = {'Cu (20C)': rho_Cu, 'Cu (77K, calc.)': rho_Cu_77K,
                      'Cu (77K, optimistic)': rho_Cu_77K_approx, 'S.C.': rho_SC}
res_keys = sorted(resistivities_dict.keys())

# color coding for regions
region_key = html.Div([html.Span("PS1, ", style={"color": "rgba(0, 256, 0, 1.0)"}),
              html.Span("PS2, ", style={"color": "rgba(245, 200, 0, 1.0)"}),
              html.Span("TS1", style={"color": "rgba(245, 0, 227, 1.0)"})])

# load nominal PS geom
# paramdir = '/home/ckampa/coding/helicalc/dev/params/'
# paramdir = helicalc_dir + 'dev/params/'
# paramfile = 'Mu2e_V13'
paramdir = helicalc_dir+'scripts/SolCalc_GUI/params/'
paramfile = 'Mu2eII_Dev'

#df_PS_nom = read_solenoid_geom_combined(paramdir, paramfile).iloc[:3]
# updated to include different conductor geometries
df_PS_nom = read_solenoid_geom_combined(paramdir, paramfile, parse_config_name=True)
conductor_configs = np.unique(df_PS_nom.conductor_config)

# calculate layer thickness
# FIXME!

# integration params
drz = np.array([5e-3, 1e-2])

# editable vs. dependent columns
cols_edit_glob = ['Coil_Num', 'Ri', 'z',  'N_layers',
             'N_turns', 'I_turn',]
cols_edit_cond_glob = ['h_cable', 'w_cable', 'h_sc', 'w_sc', 't_gi', 't_ci', 't_il',]
cols_stat_glob = ['Ro', 'L', 'I_tot', 'N_turns_tot', 'helicity', 'phi0_deg', 'phi1_deg',
             'pitch', 'x', 'y', 'rot0', 'rot1', 'rot2']
cols_edit_cond_all = ['Coil_Num'] + cols_edit_cond_glob
cols_stat_all = ['Coil_Num'] + cols_stat_glob

# load TS+DS contribution to PS
#PSoff_file = '/home/shared_data/Bmaps/SolCalc_complete/Mu2e_V13.SolCalc.PS_region.standard.PSoff.pkl'
PSoff_file = helicalc_data+'Bmaps/aux/Mu2e_V13.SolCalc.PS_region.standard.PSoff.pkl'
df_PSoff = pd.read_pickle(PSoff_file)
df_PSoff = df_PSoff.astype(float)
# m = (df_PSoff.Y == 0.) & (np.isin(df_PSoff.X - 3.904, [0., 0.4, 0.7]))
#m = (df_PSoff.Y == 0.) & (np.isin(df_PSoff.X, [3.904, 4.304, 4.604]))
m = (df_PSoff.Y == 0.) & (np.isin(df_PSoff.X, [3.904, 4.054, 4.404]))
df_PSoff_lines = df_PSoff[m].copy().reset_index(drop=True, inplace=False)
# print(df_PSoff_lines)

# formatting/style
green = 'rgb(159, 210, 128)'
plot_bg = 'rgb(240, 240, 240)'

button_style = {'fontSize': 'large',
                'backgroundColor': green,
                }

# plot globals
# marker_size = 10
marker_size = 3
fsize_plot = 20
fsize_ticks = 14

# instantiate app
app = dash.Dash(name='solcalc', external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1('SolCalc Magnet Builder (Production Solenoid)'),
    html.H2('Coils Plot'),
    html.Details([
        html.Summary(''),
        dcc.Graph(id='coils-plot'),
        ], open=False,),
    html.H2('Coil Geometries'),
    html.H3('Diagram'),
    html.Details([
        html.Summary(''),
                 ], open=False),
    # ADD DIAGRAM HERE
    # tables
    html.H3('Editable Parameters'),
    html.Details([
        html.Summary(''),
        html.H4('Initialize Conductor:'),
        dcc.Dropdown(
            id='conductor-select',
            options=conductor_configs,
            value='Mu2e_PS_NbTi',
            multi=False,
            #style=desc_style,
        ),
        html.H4('Coil Parameters:'),
        dash_table.DataTable(id='editable-table',
                             columns=[{'name':i, 'id': i, 'hideable':True, 'type':'numeric',
                             'format': Format(scheme=Scheme.fixed, precision=4),} for i in cols_edit_glob],
                             data=df_PS_nom.query('conductor_config == "Mu2e_PS_NbTi"')[cols_edit_glob].to_dict('records'),
                             editable=True, row_deletable=True),
        html.Button('Add Coil', id='editable-table-button', n_clicks=0),
        html.H4('Conductor Parameters:'),
        dash_table.DataTable(id='editable-table-cond',
                             columns=[{'name':i, 'id': i, 'hideable':True, 'type':'numeric',
                             'format': Format(scheme=Scheme.fixed, precision=4),} for i in cols_edit_cond_all],
                             data=df_PS_nom.query('conductor_config == "Mu2e_PS_NbTi"')[cols_edit_cond_all].to_dict('records'),
                             editable=True, row_deletable=False),
        ], open=False,),
    html.Br(),
    html.Button('Recalculate Field', id='calc-button', style=button_style),
    html.H3('Static / Dependent Parameters:'),
    html.Details([
        html.Summary(''),
        # FIXME!
        # not positive best placement for these
        html.H3('Static/Dependent Parameters'),
        dash_table.DataTable(id='static-table',
                             columns=[{'name':i, 'id': i, 'hideable':True, 'type':'numeric',
                             'format': Format(scheme=Scheme.fixed, precision=4),} for i in cols_stat_all],
                             data=df_PS_nom.query('conductor_config == "Mu2e_PS_NbTi"')[cols_stat_all].to_dict('records'),
                             editable=False),
        html.H3('Notes on Dependent Parameters'),
        # dcc.Markdown('''
        # $R_o = R_i + h_{cable}*N_{layers} + 2*t_{gi} + 2*t_{ci}*N_{layers} + 2*{t_il}*(N_{layers}-1)$
        # '''),
        #html.Div(html.P(['Notes on depdendent parameters:', html.Br(),
        html.Div(html.P([
                        'Ro = Ri + h_cable*N_layers + 2*t_gi + 2*t_ci*N_layers + t_il*(N_layers-1) [note nominal is slightly larger by ~1 mm]', html.Br(),
                        'pitch = h_cable + 2*t_ci', html.Br(),
                        'L = pitch*N_turns + 2*t_gi [note nominal seems to use (N_turns-1)]', html.Br(),
                        'N_turns_tot = N_turns * N_layers', html.Br(),
                        'I_tot = I_turn * N_turns_tot',])),
        ], open=False),
    # field requiremetns
    html.Br(),
    html.H2('Field Requirements'),
    html.Div(children='(See mu2e-docdb-#s: 945, 947, 1266)'),
    html.Label('Plotting Options:'),
    html.Label('Axial Variable:'),
    dcc.RadioItems(
        id='x-var-req',
        options=[{'label': i, 'value': i} for i in ['z', 's']],
        value='z',
        labelStyle={'display': 'inline-block'},
        #style=desc_style,
    ),
    # SUMMARY OF WHETHER REQ SATISFIED HERE
    html.H3('Axial Field Values'),
    # region_key,
    html.Details([
        html.Summary(''),
        dcc.Graph(id='axial-field-plot'),
        ], open=True),
    html.H3('Field Gradients'),
    # region_key,
    html.Details([
        html.Summary(''),
        dcc.Graph(id='gradient-field-plot'),
        ], open=True),
    # cable length and power
    html.Br(),
    html.H2('Cable Length & Resistive Power Consumption'),
    html.Details([
        html.Summary(''),
        html.Div(children='(Click "Recalculate Field" to update)'),
        # html.Label('Total Length [m]:'),
        html.H3('Length'),
        html.Div(id='tot-length-out'),
        html.Div(id='lengths-out'),
        # html.Label('Power:'),
        html.H3('Power'),
        html.Label('Select Resistivity [Ohm m]:'),
        dcc.Dropdown(
            id='resistivity',
            options=res_keys,
            value='S.C.',
            multi=False,
            #style=desc_style,
        ),
        html.Div(id='resistivity-out'),
        # html.Label('Total Power [MW]:'),
        html.Div(id='tot-power-out'),
        html.Div(id='powers-out'),
        ], open=False,),
    # field plot
    html.H2('Field Plot'),
    html.H3('Plotting Options:'),
    html.Details([
        html.Summary(''),
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
            options=[{'label': i, 'value': i} for i in ['Tesla', 'Gauss']],
            value='Tesla',
            labelStyle={'display': 'inline-block'},
            #style=desc_style,
        ),
        ], open=False,),
    dcc.Graph(id='field-plot'),
    # hidden divs for data
    html.Div(children=df_PS_nom.query('conductor_config == "Mu2e_PS_NbTi"')[cols_edit_glob+cols_edit_cond_glob+cols_stat_glob].to_json(),
             id='geom-data', style={'display': 'none'}),
    html.Div(id='field-data', style={'display': 'none'}),
])

# conductor selection
# @app.callback(
#     [Output('editable-table', 'data'),
#      Output('editable-table-cond', 'data'),
#      Output('calc-button', 'n_clicks'),],
#     [Input('conductor-select', 'value')],
#     [State('calc-button', 'n_clicks')],
# )
# def initialize_params(conductor_key, n_clicks):
    # df_ = df_PS_nom.query(f'conductor_config == "{conductor_key}"')
    # df_edit = df_[cols_edit]
    # df_edit_cond = df_[cols_edit_cond]
    # if n_clicks is None:
    #     n_clicks = 0
    # return df_edit.to_dict('records'), df_edit_cond.to_dict('records'), n_clicks+1

# # delete rows
# @app.callback(
#     [Output('editable-table-cond', 'data'),
#      Output('static-table', 'data'),],
#     [Input('editable-table', 'data')],
#     [State('editable-table-cond', 'data'),
#      State('static-cond', 'data'),]
# )
# def update_rows(rows_edit, rows_edit_cond, rows_stat):
#     # available coil_nums
#     CNs = [r['Coil_Num'] for r in rows_edit]
#     # loop through to check for any that should be removed
#     rows_edit_cond_updated = []
#     for row in rows_edit_cond:
#         if row['Coil_Num'] in CNs:
#             rows_edit_cond_updated.append(row)
#     rows_stat_updated = []
#     for row in rows_stat:
#         if row['Coil_Num'] in CNs:
#             rows_stat_updated.append(row)
#     # FIXME! loop through to check for any that should be added

#     return rows_edit_cond_updated, rows_stat_updated


# update geom div when button is clicked
@app.callback(
    [Output('geom-data', 'children'),
     Output('static-table', 'data'),
     Output('editable-table-cond', 'data'),
     Output('editable-table', 'data')],
    [Input('calc-button', 'n_clicks'),
     Input('editable-table-button', 'n_clicks'),
     Input('editable-table', 'data_timestamp'),
     Input('conductor-select', 'value'),],
    [State('static-table', 'data'),
     State('static-table', 'columns'),
     State('editable-table', 'data'),
     State('editable-table', 'columns'),
     State('editable-table-cond', 'data'),
     State('editable-table-cond', 'columns')],
)
def update_geom_data(n_clicks_calc, n_clicks_edit, edit_timestamp, conductor_key, rows_stat, cols_stat, rows_edit, cols_edit, rows_edit_cond, cols_edit_cond):
    # update depends on context!
    # print(ctx.triggered_id)
    if ctx.triggered_id == 'conductor-select':
        ## update if a different conductor key is selected
        df_ = df_PS_nom.query(f'conductor_config == "{conductor_key}"')
        df_edit = df_[[c['name'] for c in cols_edit]]
        df_edit_cond = df_[[c['name'] for c in cols_edit_cond]]
        df_stat = df_[[c['name'] for c in cols_stat]]
    else:
        if (ctx.triggered_id == 'editable-table') or (ctx.triggered_id == 'editable-table-button'):
            # available coil_nums
            CNs = [r['Coil_Num'] for r in rows_edit]
            # add row, filled with previous row
            if ctx.triggered_id == 'editable-table-button':
                if n_clicks_edit > 0:
                    new_row = {c['id']: rows_edit[-1][c['name']] if c['name'] != 'Coil_Num' else rows_edit[-1][c['name']]+1 for c in cols_edit}
                    rows_edit.append(new_row)
                    CNs.append(CNs[-1] + 1)
            # First check for consistent rows
            # loop through to check for any that should be removed
            rows_edit_cond_updated = []
            for row in rows_edit_cond:
                if row['Coil_Num'] in CNs:
                    rows_edit_cond_updated.append(row)
            rows_stat_updated = []
            for row in rows_stat:
                if row['Coil_Num'] in CNs:
                    rows_stat_updated.append(row)
            # then check for any rows that need to be added
            CNs_stat = [r['Coil_Num'] for r in rows_stat]
            CNs_cond = [r['Coil_Num'] for r in rows_edit_cond]
            for CN in CNs:
                if CN not in CNs_stat:
                    new_row_stat = {c['id']: rows_stat[-1][c['name']] if c['name'] != 'Coil_Num' else CN for c in cols_stat}
                    rows_stat_updated.append(new_row_stat)
                if CN not in CNs_cond:
                    new_row_cond = {c['id']: rows_edit_cond[-1][c['name']] if c['name'] != 'Coil_Num' else CN for c in cols_edit_cond}
                    rows_edit_cond_updated.append(new_row_cond)
            # replace row objects
            rows_edit_cond = rows_edit_cond_updated
            rows_stat = rows_stat_updated
        print(rows_edit)
        # FIXME! A lot cleaner to concat the dataframes here. Then I don't need to track which parameter exists where...
        # load data
        df_edit = pd.DataFrame(rows_edit, columns=[c['name'] for c in cols_edit], dtype=float)
        df_edit_cond = pd.DataFrame(rows_edit_cond, columns=[c['name'] for c in cols_edit_cond], dtype=float)
        # print(df_edit)
        # print(df_edit.info())
        df_stat = pd.DataFrame(rows_stat, columns=[c['name'] for c in cols_stat], dtype=float)
    # calculations
    if 'w_cable' in df_stat.columns:
        df_ = df_stat
    else:
        df_ = df_edit_cond
    # recalculate pitch, otherwise L will be wrong -- needed once I change conductor params
    df_stat.loc[:, 'pitch'] = df_.loc[:, 'w_cable'] + 2*df_.loc[:, 't_ci']
    # I think correct!
    df_stat.loc[:, 'Ro'] = df_edit.Ri + df_.h_cable * df_edit.N_layers + \
    2 * df_.t_gi + 2*df_.t_ci*df_edit.N_layers +\
    df_.t_il*(df_edit.N_layers - 1)
    # no ground insulation
    # only one interlayer, not 2
    # df_stat.loc[:, 'Ro'] = df_edit.Ri + df_.h_cable * df_edit.N_layers + \
    # 2*df_.t_ci*df_edit.N_layers +\
    # df_.t_il*(df_edit.N_layers - 1)
    # I think correct!
    # df_stat.loc[:, 'L'] = df_stat.pitch * df_edit.N_turns + 2 * df_.t_gi
    # nominal seems to remove a turn...
    df_stat.loc[:, 'L'] = df_stat.pitch * (df_edit.N_turns-1) + 2 * df_.t_gi
    df_stat.loc[:, 'N_turns_tot'] = df_edit.N_turns * df_edit.N_layers
    df_stat.loc[:, 'I_tot'] = df_edit.I_turn + df_stat.N_turns_tot
    # combine results
    df_stat.reset_index(drop=True, inplace=True)
    df_edit.reset_index(drop=True, inplace=True)
    df_edit_cond.reset_index(drop=True, inplace=True)
    df = pd.concat([df_stat[cols_stat_glob], df_edit[cols_edit_glob], df_edit_cond[cols_edit_cond_glob]], axis=1, ignore_index=False)
    print(df)
    return df.to_json(), df_stat.to_dict('records'), df_edit_cond.to_dict('records'), df_edit.to_dict('records')

# update coils plot
@app.callback(
    Output('coils-plot', 'figure'),
    [Input('geom-data', 'children'),],
)
def plot_coils(df):
    df = pd.read_json(df)
    # get cylinders PS
    xs, ys, zs, cs = get_thick_cylinders_padded(df, df.Coil_Num)
    # get cylinders nominal PS
    xs_n, ys_n, zs_n, cs_n = get_thick_cylinders_padded(df_PS_nom.query('conductor_config == "Mu2e_PS_NbTi"'), [1, 2, 3])
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
    # print(df_calc)
    # print(df_calc.info())
    return df_calc.to_json()

# lengths and power
@app.callback(
    [Output('tot-length-out', 'children'),
     Output('lengths-out', 'children'),
     Output('resistivity-out', 'children'),
     Output('tot-power-out', 'children'),
     Output('powers-out', 'children'),],
    [Input('geom-data', 'children'),
     Input('resistivity', 'value'),],
)
def length_and_power(df, res_key):
    geom_df = pd.read_json(df)
    rho = resistivities_dict[res_key]
    tup = calc_resistive_power_coils(geom_df, resistivity=rho, full_cable=False)
    power_tot_MW, _, power_list_MW, _, R_cables, R_list, I_list, \
    L_cable_coils, L_cable_list = tup
    # reformat for nicer printing
    L_cable_coils = f'Total Length: {L_cable_coils:0.2e} [m]'
    L_cable_list = 'Coil Lengths: '+', '.join([f'{i+1}: {L:0.2e} [m]' for i, L in enumerate(L_cable_list)])
    rho = f'rho = {rho:0.2e} [Ohm m]'
    power_tot_MW = f'Total Power: {power_tot_MW:0.2f} [MW]'
    power_list_MW = 'Coil Power:'+', '.join([f'{i+1}: {P:0.3f} [MW]' for i, P in enumerate(power_list_MW)])
    return L_cable_coils, L_cable_list, rho, power_tot_MW, power_list_MW

# update axial field plot (requirements)
@app.callback(
    Output('axial-field-plot', 'figure'),
    [Input('field-data', 'children'),
     Input('x-var-req', 'value'),],
)
def axial_field_plot(df, xvar):
    df = pd.read_json(df).astype(float)
    zs = df.Z.values
    xs = np.sort(df.X.unique())[:1]
    rs = xs - 3.904
    # shared calculations
    if xvar == 'z':
        x_func = lambda z: z
        # zs = df.Z.values
        # zs_PS1 = PS1_z_range
        # zs_PS2 = PS2_z_range
        # zs_TS1 = TS1_z_range
    else:
        x_func = lambda z: s_PS(z)
        # zs = s_PS(df.Z.values)
        # zs_PS1 = [s_PS(z_) for z_ in PS1_z_range]
        # zs_PS2 = [s_PS(z_) for z_ in PS2_z_range]
        # zs_TS1 = [s_PS(z_) for z_ in TS1_z_range]
    # grab axial values
    ycol = 'Bz' # field
    m1 = (df.X == xs[0]) # axial line
    ms = [m1,]
    # plot axial trace
    B = df[ycol].values.astype(float)
    # add TS contribution
    B += df_PSoff_lines[ycol].values
    # convert to Tesla
    unit = 'Tesla'
    B *= 1e-4
    t_inc = ''
    # start trace collection
    # data = []
    # main data
    Bz_traces = [go.Scatter(x=x_func(zs[m_]), y=B[m_], mode='lines+markers',
                            marker={'color':c, 'size': marker_size, 'opacity': 0.85,
                            'line': {'width':0.1, 'color': 'white'}},
                            line={'width':1, 'color': c},
                            name=f'R = {r:0.2f}'
                            ) for m_, c, r in zip(ms, ['blue',], rs)]
    # plot requirements related things
    # point values
    PS1_Bz_trace1 = go.Scatter(x=[x_func(z_PS1_Bz_min), ], y=[PS1_Bz_min + 0.2,],
                               name=f'Minimum Field @ s={s_PS1_Bz_min:0.2f} m<br>Bz = {PS1_Bz_min:0.2f} T',
                               mode='markers', marker=dict(color='black', size=1.0,),
                               error_y=dict(
                                   width=10,
                                   thickness=1.5,
                                   color='black',
                                   type='data',
                                   symmetric=True,
                                   array=[0.2],)
                               )
    PS2_Bz_trace1 = go.Scatter(x=[x_func(z_PS2_Bz_nom),], y=[PS2_Bz_nom,],
                               name=f'Allowed Field @ s={s_PS2_Bz_nom:0.2f} m<br>Bz = [{PS2_Bz_nom*(1-tol_PS2_Bz):0.3f}, {PS2_Bz_nom*(1+tol_PS2_Bz):0.3f}] T',
                               mode='markers', marker=dict(color='rgba(130,0,255,1)', size=1.0,),
                               error_y=dict(
                                   width=10,
                                   thickness=1.5,
                                   color='rgba(130,0,255,1)',
                                   type='data',
                                   symmetric=True,
                                   array=[PS2_Bz_nom * tol_PS2_Bz,],)
                               )
    TS1_Bz_trace1 = go.Scatter(x=[x_func(z_TS1_Bz_nom),], y=[TS1_Bz_nom,],
                               name=f'Allowed Field @ s={s_TS1_Bz_nom:0.2f} m<br>Bz = [{TS1_Bz_nom*(1-tol_TS1_Bz):0.3f}, {TS1_Bz_nom*(1+tol_TS1_Bz):0.3f}] T',
                               mode='markers', marker=dict(color='rgba(162,48,48,1)', size=1.0,),
                               error_y=dict(
                                   width=10,
                                   thickness=1.5,
                                   color='rgba(162,48,48,1)',
                                   type='data',
                                   symmetric=True,
                                   array=[TS1_Bz_nom * tol_TS1_Bz,],)
                               )
    # ranges
    # print(df)
    # add TS
    df.loc[:, 'Bz'] = df.loc[:, 'Bz'] + df_PSoff_lines.loc[:, 'Bz']
    df0 = df.query(f'X == {xs[0]}').copy()
    df0.loc[:, 'Bz'] = df0.loc[:, 'Bz']*1e-4
    tup = check_PS_axial_values(df0, z_col='Z', Bz_col='Bz', x0=xs[0], y0=0.,
                          PS1_z_range=PS1_z_range, PS2_z_range=PS2_z_range,
                          dZ=0.001, z_at_min=z_PS2_Bz_nom,
                          z_at_max=PS2_z_range[0], z_PS1_Bz_min=z_PS1_Bz_min,
                          PS1_Bz_min=PS1_Bz_min, z_PS2_Bz_nom=z_PS2_Bz_nom,
                          PS2_Bz_nom=PS2_Bz_nom, tol_PS2_Bz=tol_PS2_Bz,
                          z_TS1_Bz_nom=z_TS1_Bz_nom, TS1_Bz_nom=TS1_Bz_nom,
                          tol_TS1_Bz=tol_TS1_Bz)
    to_spec, N_out_of_tol, map_in_tol, zs_interp_PS2, Bnom_interp, Bnom_interp_up, Bnom_interp_down, \
    Bzmax_val_to_spec, Bmax, Bzmax_loc_to_spec, zmax_actual, Bz_at_PS1_loc_to_spec, Bz_PS1, \
    Bz_PS2_TS1_to_spec, Bz_PS2_TS1, Bz_TS1_TS2_to_spec, Bz_TS1_TS2 = tup
    # add max value as different color scatter point
    if Bzmax_loc_to_spec:
        c='green'
        sym='diamond'
    else:
        c='red'
        sym='cross'
    Bz_traces.append(go.Scatter(x=[x_func(zmax_actual)], y=[Bmax], mode='markers',
                            marker={'color':c, 'size': 10, 'opacity': 0.85,
                            'line': {'width':0.1, 'color': 'white'}, 'symbol':sym},
                            line={'width':1, 'color': 'white'},
                            name=f'B{xvar}_max={Bmax:0.2f} T @ {xvar}={x_func(zmax_actual):0.3f} m<br>'+
                            f'In PS1? {Bzmax_loc_to_spec}'
                            ))
    # print(to_spec, Bzmax_val_to_spec, Bzmax_loc_to_spec, Bz_at_PS1_loc_to_spec, Bz_PS2_TS1_to_spec, Bz_TS1_TS2_to_spec)
    # print(f'Bzmax_PS1_loc: {Bz_PS1, PS1_Bz_min}')
    # lines
    TS_allow_trace1 = go.Scatter(x=x_func(zs_interp_PS2), y=Bnom_interp_down,
                                 mode='lines', line=dict(dash='dash', color='gray'),
                                 name='Allowed Range (PS2)', legendgroup='PS2_range',
                                )
    TS_allow_trace2 = go.Scatter(x=x_func(zs_interp_PS2), y=Bnom_interp_up,
                                 mode='lines', line=dict(dash='dash', color='gray'),
                                 name='Allowed Range (PS2)', legendgroup='PS2_range',
                                 showlegend=False,
                                )
    # region annotations
    PS1_annot = go.layout.Annotation(dict(
                        x=x_func(PS1_z_range[0]+0.2),
                        y=0.1,
                        ax=x_func(PS1_z_range[0]+0.2),
                        ay=0.1,
                        axref='x',
                        ayref='y',
                        showarrow=False,
                        text='PS1',
                        font=dict(color='black', size=16,))
    )
    PS2_annot = go.layout.Annotation(dict(
                        x=x_func(PS2_z_range[0]+0.2),
                        y=0.1,
                        ax=x_func(PS2_z_range[0]+0.2),
                        ay=0.1,
                        axref='x',
                        ayref='y',
                        showarrow=False,
                        text='PS2',
                        font=dict(color='black', size=16,))
    )
    TS1_annot = go.layout.Annotation(dict(
                        x=x_func(TS1_z_range[0]+0.2),
                        y=0.1,
                        ax=x_func(TS1_z_range[0]+0.2),
                        ay=0.1,
                        axref='x',
                        ayref='y',
                        showarrow=False,
                        text='TS1',
                        font=dict(color='black', size=16,))
    )
    # PS1_Bz_annot = go.layout.Annotation(dict(
    #                        x=x_func(z_PS1_Bz_min),
    #                        y=PS1_Bz_min + 0.2,
    #                        ax=x_func(z_PS1_Bz_min),
    #                        ay=PS1_Bz_min,
    #                        text=f'f<br><br><br><br>Minimum Field @ s={s_PS1_Bz_min:0.2f} m<br>Bz = {PS1_Bz_min:0.2f} T<br><br><br><br>f',
    #                        font=dict(size=10),
    #                        showarrow=True,
    #                        axref='x',
    #                        ayref='y',
    #                        arrowhead=3,
    #                        arrowwidth=1.5,
    #                        arrowcolor='rgba(0,0,0,0.8)',)
    # )
    # PS2/TS1 value (range)
    ####
    # TS1/TS2 value (range)
    ####
    # TS2 values / gradient (region)
    ####
    # add all traces to layout
    #data = [PS1_trace, PS2_trace, TS1_trace, PS1_Bz_trace,] + Bz_traces
    data = [PS1_Bz_trace1, PS2_Bz_trace1, TS1_Bz_trace1, TS_allow_trace1, TS_allow_trace2] + Bz_traces
    # layout should work with all configurations
    layout = go.Layout(
        title=f'PS Axis Field Requirements (x==3.904, y==0.0) m',
        height=700,
        font=dict(family="Courier New", size=fsize_plot,),
        margin={'l': 60, 'b': 60, 't': 60, 'r': 60},
        scene=dict(aspectmode='auto',
                   #xaxis={'title': 'Z [m]', 'tickfont':{'size': fsize_ticks}},
                   #yaxis={'title': f'{ycol} [{unit}]', 'tickfont':{'size': fsize_ticks}},
        ),
        xaxis={'title': f'{xvar} [m]', 'tickfont':{'size': fsize_ticks}},
        yaxis={'title': f'B{xvar} [T]', 'tickfont':{'size': fsize_ticks}},
        plot_bgcolor=plot_bg,
        showlegend=True,
        annotations=[PS1_annot, PS2_annot, TS1_annot,],
    )

    fig = go.Figure(data=data, layout=layout)

    # add regions
    # PS1
    fig.add_vrect(x0=x_func(PS1_z_range[0]), x1=x_func(PS1_z_range[1]),
                  fillcolor='rgba(0, 256, 0, 0.1)', layer='below', line_width=0,)
    # PS2
    fig.add_vrect(x0=x_func(PS2_z_range[0]), x1=x_func(PS2_z_range[1]),
                  fillcolor='rgba(245, 200, 0, 0.1)', layer='below', line_width=0,)
    # TS1
    fig.add_vrect(x0=x_func(TS1_z_range[0]), x1=x_func(TS1_z_range[1]),
                  fillcolor='rgba(245, 0, 227, 0.1)', layer='below', line_width=0,)

    return fig
    # return {'data': fig.data, 'layout': fig.layout}
    # return {'data':data, 'layout':layout}

# update gradient field plot (requirements)
@app.callback(
    Output('gradient-field-plot', 'figure'),
    [Input('field-data', 'children'),
     Input('x-var-req', 'value'),],
)
def axial_field_plot(df, xvar):
    df = pd.read_json(df).astype(float)
    zs = df.Z.values
    xs = np.sort(df.X.unique())
    rs = xs - 3.904
    # shared calculations
    if xvar == 'z':
        x_func = lambda z: z
    else:
        x_func = lambda z: s_PS(z)
    # grab axial values
    ycol = 'Bz' # field
    m1 = (df.X == xs[0]) # axial line
    m2 = (df.X == xs[1])
    m3 = (df.X == xs[2])
    ms = [m1, m2, m3]
    # plot axial trace
    B = df[ycol].values.astype(float)
    # add TS contribution
    B += df_PSoff_lines[ycol].values
    # convert to Tesla
    unit = 'Tesla'
    B *= 1e-4
    # start trace collection
    Bz_traces = []
    # loop through lines
    for m_, c, r in zip(ms, ['blue', 'green', 'red'], rs):
        dB = np.diff(B[m_])/np.diff(x_func(zs[m_]))
        # main data
        Bz_traces.append(go.Scatter(x=x_func(zs[m_]), y=dB, mode='lines+markers',
                                    marker={'color':c, 'size': marker_size, 'opacity': 0.85,
                                    'line': {'width':0.1, 'color': 'white'}},
                                    line={'width':1, 'color': c},
                                    name=f'R = {r:0.2f}'
                                   ))
    # plot requirements related things
    # point values
    # PS1_Bz_trace1 = go.Scatter(x=[x_func(z_PS1_Bz_min), ], y=[PS1_Bz_min + 0.2,],
    #                            name=f'Minimum Field @ s={s_PS1_Bz_min:0.2f} m<br>Bz = {PS1_Bz_min:0.2f} T',
    #                            mode='markers', marker=dict(color='black', size=1.0,),
    #                            error_y=dict(
    #                                width=10,
    #                                thickness=1.5,
    #                                color='black',
    #                                type='data',
    #                                symmetric=True,
    #                                array=[0.2],)
    #                            )
    # PS2_Bz_trace1 = go.Scatter(x=[x_func(z_PS2_Bz_nom),], y=[PS2_Bz_nom,],
    #                            name=f'Allowed Field @ s={s_PS2_Bz_nom:0.2f} m<br>Bz = [{PS2_Bz_nom*(1-tol_PS2_Bz):0.3f}, {PS2_Bz_nom*(1+tol_PS2_Bz):0.3f}] T',
    #                            mode='markers', marker=dict(color='rgba(130,0,255,1)', size=1.0,),
    #                            error_y=dict(
    #                                width=10,
    #                                thickness=1.5,
    #                                color='rgba(130,0,255,1)',
    #                                type='data',
    #                                symmetric=True,
    #                                array=[PS2_Bz_nom * tol_PS2_Bz,],)
    #                            )
    # TS1_Bz_trace1 = go.Scatter(x=[x_func(z_TS1_Bz_nom),], y=[TS1_Bz_nom,],
    #                            name=f'Allowed Field @ s={s_TS1_Bz_nom:0.2f} m<br>Bz = [{TS1_Bz_nom*(1-tol_TS1_Bz):0.3f}, {TS1_Bz_nom*(1+tol_TS1_Bz):0.3f}] T',
    #                            mode='markers', marker=dict(color='rgba(162,48,48,1)', size=1.0,),
    #                            error_y=dict(
    #                                width=10,
    #                                thickness=1.5,
    #                                color='rgba(162,48,48,1)',
    #                                type='data',
    #                                symmetric=True,
    #                                array=[TS1_Bz_nom * tol_TS1_Bz,],)
    #                            )
    # ranges
    # print(df)
    # add TS
    # df.loc[:, 'Bz'] = df.loc[:, 'Bz'] + df_PSoff_lines.loc[:, 'Bz']
    # df0 = df.query(f'X == {xs[0]}').copy()
    # df0.loc[:, 'Bz'] = df0.loc[:, 'Bz']*1e-4
    # tup = check_PS_axial_values(df0, z_col='Z', Bz_col='Bz', x0=xs[0], y0=0.,
    #                       PS1_z_range=PS1_z_range, PS2_z_range=PS2_z_range,
    #                       dZ=0.001, z_at_min=z_PS2_Bz_nom,
    #                       z_at_max=PS2_z_range[0], z_PS1_Bz_min=z_PS1_Bz_min,
    #                       PS1_Bz_min=PS1_Bz_min, z_PS2_Bz_nom=z_PS2_Bz_nom,
    #                       PS2_Bz_nom=PS2_Bz_nom, tol_PS2_Bz=tol_PS2_Bz,
    #                       z_TS1_Bz_nom=z_TS1_Bz_nom, TS1_Bz_nom=TS1_Bz_nom,
    #                       tol_TS1_Bz=tol_TS1_Bz)
    # to_spec, N_out_of_tol, map_in_tol, zs_interp_PS2, Bnom_interp, Bnom_interp_up, Bnom_interp_down, \
    # Bzmax_val_to_spec, Bmax, Bzmax_loc_to_spec, zmax_actual, Bz_at_PS1_loc_to_spec, Bz_PS1, \
    # Bz_PS2_TS1_to_spec, Bz_PS2_TS1, Bz_TS1_TS2_to_spec, Bz_TS1_TS2 = tup
    # add max value as different color scatter point
    # if Bzmax_loc_to_spec:
    #     c='green'
    #     sym='diamond'
    # else:
    #     c='red'
    #     sym='cross'
    # Bz_traces.append(go.Scatter(x=[x_func(zmax_actual)], y=[Bmax], mode='markers',
    #                         marker={'color':c, 'size': 10, 'opacity': 0.85,
    #                         'line': {'width':0.1, 'color': 'white'}, 'symbol':sym},
    #                         line={'width':1, 'color': 'white'},
    #                         name=f'B{xvar}_max={Bmax:0.2f} T @ {xvar}={x_func(zmax_actual):0.3f} m<br>'+
    #                         f'In PS1? {Bzmax_loc_to_spec}'
    #                         ))
    # print(to_spec, Bzmax_val_to_spec, Bzmax_loc_to_spec, Bz_at_PS1_loc_to_spec, Bz_PS2_TS1_to_spec, Bz_TS1_TS2_to_spec)
    # print(f'Bzmax_PS1_loc: {Bz_PS1, PS1_Bz_min}')
    # lines
    # TS_allow_trace1 = go.Scatter(x=x_func(zs_interp_PS2), y=Bnom_interp_down,
    #                              mode='lines', line=dict(dash='dash', color='gray'),
    #                              name='Allowed Range (PS2)', legendgroup='PS2_range',
    #                             )
    # TS_allow_trace2 = go.Scatter(x=x_func(zs_interp_PS2), y=Bnom_interp_up,
    #                              mode='lines', line=dict(dash='dash', color='gray'),
    #                              name='Allowed Range (PS2)', legendgroup='PS2_range',
    #                              showlegend=False,
    #                             )
    # region annotations
    PS1_annot = go.layout.Annotation(dict(
                        x=x_func(PS1_z_range[0]+0.2),
                        y=0.5,
                        ax=x_func(PS1_z_range[0]+0.2),
                        ay=0.5,
                        axref='x',
                        ayref='y',
                        showarrow=False,
                        text='PS1',
                        font=dict(color='black', size=16,))
    )
    PS2_annot = go.layout.Annotation(dict(
                        x=x_func(PS2_z_range[0]+0.2),
                        y=0.5,
                        ax=x_func(PS2_z_range[0]+0.2),
                        ay=0.5,
                        axref='x',
                        ayref='y',
                        showarrow=False,
                        text='PS2',
                        font=dict(color='black', size=16,))
    )
    TS1_annot = go.layout.Annotation(dict(
                        x=x_func(TS1_z_range[0]+0.2),
                        y=0.5,
                        ax=x_func(TS1_z_range[0]+0.2),
                        ay=0.5,
                        axref='x',
                        ayref='y',
                        showarrow=False,
                        text='TS1',
                        font=dict(color='black', size=16,))
    )
    ####
    # add all traces to layout
    #data = [PS1_Bz_trace1, PS2_Bz_trace1, TS1_Bz_trace1, TS_allow_trace1, TS_allow_trace2] + Bz_traces
    data = [] + Bz_traces
    # layout should work with all configurations
    layout = go.Layout(
        title=f'PS Field Gradient Requirements (y==0.0) m',
        height=700,
        font=dict(family="Courier New", size=fsize_plot,),
        margin={'l': 60, 'b': 60, 't': 60, 'r': 60},
        scene=dict(aspectmode='auto',
                   #xaxis={'title': 'Z [m]', 'tickfont':{'size': fsize_ticks}},
                   #yaxis={'title': f'{ycol} [{unit}]', 'tickfont':{'size': fsize_ticks}},
        ),
        xaxis={'title': f'{xvar} [m]', 'tickfont':{'size': fsize_ticks}},
        yaxis={'title': f'grad_{xvar}(B{xvar}) [T/m]', 'tickfont':{'size': fsize_ticks}},
        plot_bgcolor=plot_bg,
        showlegend=True,
        annotations=[PS1_annot, PS2_annot, TS1_annot,],
    )

    fig = go.Figure(data=data, layout=layout)

    # add regions
    # PS1
    fig.add_vrect(x0=x_func(PS1_z_range[0]), x1=x_func(PS1_z_range[1]),
                  fillcolor='rgba(0, 256, 0, 0.1)', layer='below', line_width=0,)
    # PS2
    fig.add_vrect(x0=x_func(PS2_z_range[0]), x1=x_func(PS2_z_range[1]),
                  fillcolor='rgba(245, 200, 0, 0.1)', layer='below', line_width=0,)
    # TS1
    fig.add_vrect(x0=x_func(TS1_z_range[0]), x1=x_func(TS1_z_range[1]),
                  fillcolor='rgba(245, 0, 227, 0.1)', layer='below', line_width=0,)

    return fig

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
