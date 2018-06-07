function [status] = lasso_distr()

  % TODO: MPI calls not tested -- probably need to change how data is set and extracted
  % Distributed ADMM Lasso problem
  % The input data is a normally-distributed mxn random matrix
  % with m = 1000, n = 15
  % We will break the matrix row-wise into N=50 m_i x n skinny sub-matrices, 
  % where m_i = 20
  % (Later will try to break it into N=100 m_i x n fat sub-matrices,
  % where m_i = 10  -- need to use the matrix inversion lemma)

  % For each iteration we have two kinds of messages, each referencing a mesage type tag
  % Main process sends messages to the processors with the necessary info
  %   for computing the xi's.  The only vectors passed are z and the ui for the process
  % Processors do the processing, each sending a message back to the main process with
  %   a computed xi value.  
  % Main process receives the messages with xi vectors from the processors, storing
  % these vectors in an array, then averaging them along with ui vectors to get 
  % xbar and ubar and from these, the new z
  % Main process then computes an array of new u vectors, one for each processor.  
  % These are computed by adding xi - z to the previous ui vectors 
  % for each processor

  clear()
  newmat = false;
  debug = false;
  % for this implementation to make sense
  % we need a tall thin matrix.
  m = 1000;
  n = 15;
  if newmat
    A = rand(m,n);
    b = rand(m,1);
    save('LAd.mat','A')
    save('Lbd.mat','b')
  else
    load('LAd.mat')
    load('Lbd.mat')
  end
  % split the original matrix and store cholesky-factored (Ai'*Ai + rho*I)
  % matrices as well as b and u vectors for use by the p processes.
  p = 50;
  lambda = 10; % the weight on the 1 norm
  rho = 100; 
  %  
  alst = cell(p,1);
  rlst = cell(p,1);
  blst = cell(p,1);
  for i = 1:p
    l = m/p
    st = (i-1)*l + 1;
    Ai = A(st:st+l-1,:);
    rlst{i} = chol(Ai.'*Ai + rho*eye(n));
    blst{i} = b(st:st+l-1);
  end
  % these arrays are managed by the main process.  Each column is a vector 
  % corresponding to one of the processors.  This makes it easy to get averages.
  xs = zeros(n,p);
  us = zeros(n,p);
  curz = zeros(n,1);

  main_id = 0;
  processor_ids = 1:p;

  [rank,size_w]=MPI('Init');
  if debug; fprintf('\n I am %d.  MPI was initialized\n',rank); end;

  iters=50;
  if rank == main_id
    fs = zeros(iters,1);
    ns = zeros(iters,1);
  end

  % Main algorithm
  for i = 1:iters
    if rank == main_id % I'm the main process
      % I will send a message to each of the processors with the data to process.
      for j = 1:p
        destination = j;
        tag=11;
        a = [curz;us(i,:)];
        MPI('Send', a, numel(a), 'MPI_DOUBLE', destination, tag, 'MPI_COMM_WORLD');
        if debug; fprintf('Main sent tag 11 to %d \n',j); end;
      end
      for j = 1:p
        x = MPI('Recv',n_elements,'MPI_DOUBLE', source, tag,'MPI_COMM_WORLD');
        if debug; fprintf("Main received tag 12 from %d", j+1); end;
        xs(:,j) = x;
      end
      % I have all the xs for this step now - I will compute z and u  
      xm = mean(xs,2)
      um = mean(us,2)
      curz = shrinkage(xm + um,lambda/rho/p);
      us = us + xs - curz;
      nx1 = norm(xm,1)
      fs(i) = 0.5*norm(A*xm - b,2)**2 + lam * nx1
      nxs(i) = nx1
    else % I am one of the helpers
      j = rank - 1
      b = MPI('Recv',n_elements,'MPI_DOUBLE', source, tag,'MPI_COMM_WORLD');
      if debug; fprint("Rank %d received tag 11 from main\n", rank); end;
      z = b(1:n);
      u = b(n+1:n_elements);
      r = rlst{j};
      b = blst{j};
      a = alst{j};
      rt = r.';
      x = r\(rt\(r*rt*b+rho*(z-u)));
      destination = 0;
      tag=12;
      a = x;
      fprintf('I am %d and I will send to %d the vector %s \n',rank,destination,num2str(a));
      MPI('Send', a, numel(a), 'MPI_DOUBLE', destination, tag, 'MPI_COMM_WORLD');
      if debug; fprintf("Rank %d sent tag 12 to main", rank); end;
    end
  end
  if rank == 0;
    save('nxs.mat','nxs');
    save('fs.mat', 'fs' );
  end
  status = 0;
  MPI('Finalize');
end

function y = shrinkage(a, k)
  y = (abs(a-k)-abs(a+k))/2 + a;
end
