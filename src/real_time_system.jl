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

function ctl(marks::Dict)
    return html_div([
        html_h2("Inference"),
        dcc_slider(
            id="file",
            min=1, max=length(marks), value=length(marks),
            marks=marks,
        )
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
    f2m = f -> match(r"_([^\/]+).mat", f).captures[]
    marks = Dict([i=>f2m(f) for (i, f) in enumerate(files)])

    app = dash(external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"])
    app.layout = html_div([
        banner(),
        ctl(marks),
        plots(files[21], width, height),
    ])

    callback!(
        app,
        Output("plots", "children"),
        Input("file", "value"),
    ) do i
        return get_plots(files[i], width, height)
    end

    return app
end

start_real_time_system() = run_server(gen_app(), "0.0.0.0", 8080)
