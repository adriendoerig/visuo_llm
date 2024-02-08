function cvn_plot_fix(sig_data, viewz, fig_path, fig_name, title_prefix, SAVE_TYPE, max_cmap_val, extraopts)
    %% plot statistical results
    % make an image with only curvature

    if max_cmap_val == 0
        cb_bound = max([0.01, round(max(abs(sig_data(:))),2)]);
    else
        cb_bound = max_cmap_val;
    end

    curvature = zeros(1, 327684);
    Lookup = [];
    wantfig = 1;
    [curv,Lookup,curv_rgbimg] = cvnlookup('fsaverage', viewz, curvature', [], cmapsign4(256), 0, Lookup, wantfig, extraopts);
%    nan_curv = cat(3, isnan(curv),  isnan(curv),  isnan(curv));
%    % replace curv_rgbimg with nans in curv
%    curv_rgbimg(nan_curv)=nan;

    wantfig = 0;
    Lookup = [];
    rgbimg = [];
    [rawimg,Lookup,rgbimg] = cvnlookup('fsaverage', viewz, sig_data', [-cb_bound, cb_bound], cmapsign4(256), [], Lookup, wantfig, extraopts);
    % here we find where in the rgbimg of the effet we have nans (non_sig)
    nan_rawimg = cat(3, isnan(rawimg), isnan(rawimg), isnan(rawimg));
    % here we identify the sig ones.
    non_nan_rawimg = ~nan_rawimg;

    % we set the sig vertices of the curvature to 0;
    curv_rgbimg(non_nan_rawimg) = 0;
    rgbimg(nan_rawimg) = 0;

    plot_img = curv_rgbimg+rgbimg;

    imagesc(plot_img);
    axis off
%    title(strcat(title_prefix, sprintf(' group average max: %3.2f', max(sig_data(:)))));

    % add a colorbar
    axes('position', [0.375 0.035 .25 .035], 'color', 'none');
    cb_data = -cb_bound:.0002:cb_bound;
    imagesc(cb_data), colormap(cmapsign4(256))
    label = {num2str(cb_bound), num2str(0), num2str(cb_bound)};
    set(gca, 'xtick', [1, round(length(cb_data)/2), length(cb_data) ])
    set(gca, 'xticklabel', label)
    set(gca,'ytick',[])
    set(gca,'yticklabel',[])

    % save results as png
    fullfile(fig_path, fig_name)
    saveas(gcf, fullfile(fig_path, fig_name), SAVE_TYPE);
end
