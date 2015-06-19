function data = generateGraph(data, thresh)

	% load 05-ADNIts.mat
	% load 06-ADNI.mat

    if nargin < 2
        thresh = 0.5;
    end
    
    if ~isfield(data,'Xfull')
        data.Xfull = data.X;
    end
    
    data.X = data.Xfull;
    data.X(data.X<0.5) = 0; 
    
    len = 112;
	gnd = data.gnd;
	gndClass = unique(gnd);

    
	Wagg = {};
	for iClass = 1:length(gndClass)
		index = find(gnd==gndClass(iClass));

		Wbrain = {};
		W_all = zeros(len,len);
		num_fea = len*(len-1)/2;
		map = gen_map(len);        
        B = triu(ones(len,len), 1);
        
		for iPatient = 1:size(data.X,2),
			% % if we could get the brain matrix directly
            % W = subject{index(iPatient)}.correlation; 
            
            % if we have to reshape the matrix from feature vector
		    A = B'; A(B'==1) = data.X(:,iPatient); W = A + A';

            W(abs(W)<=thresh) = 0;
		    Wbrain{iPatient} = W;
		end
		%-- network topology (adjacent edges)
		Wagg{iClass} = gen_fea_net(map, Wbrain); 
	end

	data.Wnc = Wagg{1};
	data.Wmci = Wagg{2};
	data.Wad = Wagg{3};

end




function [ net ] = gen_fea_net(map, Wbrain)

	len = size(map,1);
	num_fea = len*(len-1)/2;
	net = zeros(num_fea);
	% edges on same column
	for i=1:len
	    for j=1:len
	        if j==i
	            continue;
	        end
	        for k=j+1:len
	            if k==i
	                continue;
	            end
	            net(map(i,j),map(i,k)) = comp_weight(Wbrain,i,j,k);
	        end
	    end
	end

	net = max(net, net');

end

function [ weight ] = comp_weight(Wbrain,i,j,k)

    weight = 0;
    nsp = length(Wbrain);
    for p = 1:nsp
        if Wbrain{p}(i,j)~=0 && Wbrain{p}(i,k)~=0
            weight = weight + 1;
        end
    end
    weight = weight/nsp;

end


function [ map ] = gen_map( len )
	%-- map indices between brain network to edge-dual network
	%-- len is the number of nodes in brain network
	%-- example: gen_map(4)

	num_fea = len*(len-1)/2;
	map = zeros(len);
	cnt = 0;
	for i=1:len-1
	    for j=i+1:len
	        cnt = cnt+1;
	        map(j,i)=cnt;
	    end
	end
	        
	map = max(map,map');

end

