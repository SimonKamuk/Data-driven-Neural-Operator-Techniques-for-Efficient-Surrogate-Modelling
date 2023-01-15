using PyPlot, JLD2, FileIO

function pyplot_hexbin(plot_params)
    x,y,ticks,title_string,xlabel_string,ylabel_string,filename = plot_params
    figure()
    # yscale("log")
    hexbin(x,y,yscale="log",linewidths=0.05,cmap="inferno")
    title("Loss vs. depth scale for test set")
    xlabel(xlabel_string)
    ylabel(ylabel_string)
    xticks(ticks...)
    colorbar(label="Counts")
    savefig(filename)
    close()
end

pyplot_hexbin.(values(FileIO.load("hexbin_plot_data.jld2")))
