using PyPlot, JLD2, FileIO

function pyplot_hexbin(plot_params)
    x,y,ticks,title_string,xlabel_string,ylabel_string,filename = plot_params
    figure()
    # yscale("log")

    idx = y.>0

    if occursin("time",lowercase(title_string))
        # ylim(1e-10,1e4)
        hexbin(x[idx],y[idx],yscale="log",linewidths=0.05,cmap="inferno",extent=(minimum(x),maximum(x),-8,4)) 
    else
        hexbin(x[idx],y[idx],yscale="log",linewidths=0.05,cmap="inferno")
    end
    title(title_string)
    xlabel(xlabel_string)
    ylabel(ylabel_string)
    xticks(ticks...)
    colorbar(label="Counts")
    savefig(filename)
    close()
end

pyplot_hexbin.(values(FileIO.load("hexbin_plot_data.jld2")))
