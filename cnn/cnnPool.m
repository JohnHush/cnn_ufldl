function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%     

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);

pooledFeatures = zeros(convolvedDim / poolDim, ...
        convolvedDim / poolDim, numFilters, numImages);

% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages 
%   matrix pooledFeatures, such that
%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region. 
%   
%   Use mean pooling here.

%%% YOUR CODE HERE %%


for ifilter = 1:numFilters
    for iImage = 1:numImages

        % compute the blurred Features by convolution with a all ones matrix %
        blurredFeatures = zeros( convolvedDim-poolDim+1 , convolvedDim-poolDim+1 );

        % doing the convolution %
        blur_matrix = ones( poolDim , poolDim );
        blurredFeatures = conv2( squeeze(convolvedFeatures(:, : , ifilter , iImage )) , blur_matrix , "valid" );
        blurredFeatures /= poolDim*poolDim;

        % subsample the blurredFeatures %
        for i = 1:convolvedDim/poolDim
            for j = 1:convolvedDim/poolDim
                pooledFeatures(i,j,ifilter,iImage) = blurredFeatures( (i*poolDim)-poolDim+1,(j*poolDim)-poolDim+1 );
            end
        end

    end
end

end

