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

function ctl(init_filename::String)
    return html_div([
        html_h2("Inference"),
        html_div("Enter file name: "),
        dcc_input(id="file-name", value=init_filename, type="text"),
        html_button(id="submit-button-state", children="submit", n_clicks=0),
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
    ρ, w = infer(filename)

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

function gen_app(; width=500, height=500, init_filename="SQ20_5mW.mat")
    app = dash(external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"])
    app.layout = html_div([
        banner(),
        ctl(init_filename),
        plots(init_filename, width, height),
    ])

    callback!(
        app,
        Output("plots", "children"),
        Input("submit-button-state", "n_clicks"),
        State("file-name", "value")
    ) do _, input_value
        get_plots(input_value, width, height)
    end

    return app
end

start_real_time_system() = run_server(gen_app(), "0.0.0.0", 8080)
