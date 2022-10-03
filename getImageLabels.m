function imageType = getImageLabels(imset)
    imageType = categorical(repelem({imset.Description}', ...
        [imset.Count], 1));
end