using Dash
using DashHtmlComponents
using DashCoreComponents
using PlotlyJS

export start_real_time_system

function banner()
    return html_div([
        html_h1("Real Time Quantum State Tomography"),
        html_div("Real time QST: A real time state inference system for RK Lee's lab in NTHU"),
    ])
end

function ctl()
    return html_div([
        dcc_interval(id="interval", interval=3*1000, n_intervals=0),
        html_h2("Inference"),
        dcc_radioitems(id="mode", options=[
            Dict("label"=>"Single", "value"=>"S"),
            Dict("label"=>"Continuous", "value"=>"C"),
        ], value="S"),
        html_button(id="snap", children="Snap", n_clicks=0),
    ])
end

color_config = [:colorscale=>"twilight", :cmid=>0, :zmid=>0]

function get_surface(data::Vector, width::Integer, height::Integer)
    data = PlotlyJS.surface(z=data; color_config...)
    layout = Layout(title="Wigner Function", width=width, height=height)

    return plot(data, layout)
end

function get_heatmap(data::Vector, width::Integer, height::Integer)
    data = PlotlyJS.heatmap(z=data; color_config...)
    layout = Layout(title="Wigner Function", width=width, height=height)

    return plot(data, layout)
end

function get_density_matrix(data::Vector, width::Integer, height::Integer)
    data = PlotlyJS.heatmap(z=data; color_config...)
    layout = Layout(title="Density Matrix (Real Part)", width=width, height=height)

    return plot(data, layout)
end

function get_plots(filename::String, width::Integer, height::Integer)
    isempty(filename) && (return []) # TODO: handle non-exist file
    ρ, w = infer(filename, fix_θ=false)

    return [
        dcc_graph(figure=get_surface(w, width, height)),
        dcc_graph(figure=get_heatmap(w, width, height)),
        dcc_graph(figure=get_density_matrix(ρ, width, height))
    ]
end

function plots(init_filename::String, width::Integer, height::Integer)
    return html_div(
        id="plots",
        style=Dict("columnCount"=>3),
        get_plots(init_filename, width, height)
    )
end

function gen_app(; width=500, height=500)
    files = readdir(joinpath(data_path(), "Flow"))

    app = dash(external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"])
    app.layout = html_div([
        banner(),
        ctl(),
        plots(files[21], width, height),
    ])

    callback!(
        app,
        Output("plots", "children"),
        Input("snap", "n_clicks"),
    ) do n
        return get_plots(files[(n-1) % length(files) + 1], width, height)
    end

    callback!(
        app,
        Output("snap", "n_clicks"),
        Input("interval", "n_intervals"),
        State("mode", "value")
    ) do n, mode
        (mode=="S") && (return no_update())

        return n
    end

    return app
end

start_real_time_system() = run_server(gen_app(), "0.0.0.0", 8080)
