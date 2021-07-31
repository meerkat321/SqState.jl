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

function get_data()
    data = rand(10, 10)

    return [data[:, i] for i in 1:size(data, 2)]
end

color_config = [:colorscale=>"twilight", :cmid=>0, :zmid=>0]

function get_surface(width::Integer, height::Integer)
    data = PlotlyJS.surface(z=get_data(); color_config...)
    layout = Layout(width=width, height=height)

    return plot(data, layout)
end

function get_density_matrix(width::Integer, height::Integer)
    data = PlotlyJS.heatmap(z=get_data(); color_config...)
    layout = Layout(width=width, height=height)

    return plot(data, layout)
end

function plots(width::Integer, height::Integer)
    return html_div(
        style=Dict("columnCount"=>2),
        [
            dcc_graph(id="surface-graph", figure=get_surface(width, height)),
            dcc_graph(id="density-matrix-graph", figure=get_density_matrix(width, height),)
        ]
    )
end

function gen_app()
    app = dash(external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"])
    app.layout = html_div([
        banner(),
        plots(700, 700)
    ])

    return app
end

start_real_time_system() = run_server(gen_app(), "0.0.0.0", 8080)
