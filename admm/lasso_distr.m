function [status] = lasso_distr()

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
  newmat = true;
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
  p = 50
  lambda = 1
  rho = 100;
  %  
  rlst = cell(p,1);
  blst = cell(p,1);
  ulst = cell(p,1);
  for i = 1:p
    st = (i-1)*m/p + 1
    Ai = A(st:st+m/p,:)
    rlst{i} = chol(Ai.'*Ai + rho*eye(n))
    blst{i} = b(st:st+m/p)
  end
  % these arrays are managed by the main process.  Each column is a vector 
  % corresponding to one of the processors.  This makes it easy to get averages.
  xs = zeros(n,p) 
  us = zeros(n,p)
  curz = zeros(n,1);

  main_id = 0
  processor_ids = 1:p

  disp('Initializing MPI \n')
  [rank,size_w]=MPI('Init');
  fprintf('\n I am %d MPI was initialized\n',rank);

  iters=500;
  ns = zeros(iters,1);
  for i = 1:iters
    %-----------------------------------------------------
    % send/receive messages regarding: compute x given R, b
    %----------------------------------------------------- 
    if rank == 0 % I'm the main process
      % I will send a message to each of the processors with the data to process.
      for j = 1:p
        destination = j
        tag=201;
        a = [curz;us(i,:)]
        fprintf('I am %d and I will send to %d the vector %s \n',rank,destination,num2str(a));
        MPI('Send', a, numel(a), 'MPI_DOUBLE', destination, tag, 'MPI_COMM_WORLD');
      end
      received = zeros(p,1)
      while !all(received)
        fprintf('I am %d, I am going to receive \n', rank);
        b = MPI('Recv',n_elements,'MPI_DOUBLE', source, tag,'MPI_COMM_WORLD');
        fprintf('I am %d and I received the vector %s from %d \n',rank,num2str(a),source);
        xs(source,:) = x
        received(source) = 1
      end
      % I have all the xs for this step now - I will compute z and u  
      z = shrinkage(mean(xs,2) + mean(us,2),lambda/rho);
      us = us + xs - z;
    else % I am one of the processor guys
      received = false
      while !received
        fprintf('I am %d, I am going to receive \n', rank);
        b = MPI('Recv',n_elements,'MPI_DOUBLE', source, tag,'MPI_COMM_WORLD');
        z = b[1:n]
        u = b[n+1:] 
        r = rlst{j}
        rt = r.'
        b = blst{j}
        x = r\(rt\(r*rt*b+rho*(z-u)))
        destination = 0
        tag=202;
        a = x
        fprintf('I am %d and I will send to %d the vector %s \n',rank,destination,num2str(a));
        MPI('Send', a, numel(a), 'MPI_DOUBLE', destination, tag, 'MPI_COMM_WORLD');
        received = true
      end
    end
  end
  MPI('Finalize');
  fprintf('\n rank %d : End \n', rank);
  if rank == 0
    xresult = mean(xs,2)
    save('xresult.mat','xresult')
  end
  status = 0
end

function y = shrinkage(a, k)
  y = (abs(a-k)-abs(a+k))/2 + a;
end
