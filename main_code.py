import numpy as np
import random as rand
import networkx as nx
import matplotlib.pyplot as plt
from scipy import special
import time
#### bases ##################################################################################################################

start_time = time.time()

def base2number(x,b):
  # "numbers" are stored as lists, MSB first
  # compute the number x in base b
  number = 0
  q = 1
  for k in reversed(x):
    number += k*q
    q *= b
  return(number)

def number2base(x,b,length=None):
  y = []
  while x:
    y.append(int(x%b))
    x //= b
  if length != None:
    y.extend([0]*(length-len(y)))
  return(y[::-1]) # reverse output vector to MSB -> LSB

def base2base(x,b1,b2):
  number = base2number(x,b1)
  return(number2base(number, b2))

def conv_encode_symbol(input_symbol, binary_polynomials, state=None):
    if state is None:
        state = [0] * (len(binary_polynomials[0]) - 1)  # Initialize state
    
    feedback_poly = binary_polynomials[0][1:]  # Ignore leading coefficient
    feedback_bit = (input_symbol + sum(x * y for x, y in zip(state, feedback_poly))) % 2  # Compute feedback

    new_state = [feedback_bit] + state[:-1]  # Shift register update
    output_bit = sum(x * y for x, y in zip(new_state, binary_polynomials[1])) % 2  # Apply second polynomial
    
    return [input_symbol, output_bit], new_state  # Return systematic bit and encoded bit


##################### conv encoding###################################################################################


def conv_encode_list(input_list, octal_polynomials, terminated=True):
    # Convert octal polynomials to binary
    binary_polynomials = [base2base(p, 8, 2) for p in octal_polynomials]

    # Initialize encoding
    output, state = conv_encode_symbol(input_list[0], binary_polynomials)
    
    # Encode rest of the input sequence
    for input_symbol in input_list[1:]:
        new_output, state = conv_encode_symbol(input_symbol, binary_polynomials, state)
        output.extend(new_output)

    # Modified termination process
    if terminated:
        feedback_poly = binary_polynomials[0][1:]  # Feedback polynomial (ignoring leading coefficient)
        
        # Compute required termination bits
        for _ in range(len(state)):
            # Solve for input_bit: input_bit + sum(state * feedback_poly) = 0 mod 2
            termination_bit = sum(x * y for x, y in zip(state, feedback_poly)) % 2
            new_output, state = conv_encode_symbol(termination_bit, binary_polynomials, state)
            output.extend(new_output)

    return output

data_length = 12
# conv code polynomials
p1 = [1,5]
p2 = [1,7]
data = rand.getrandbits(data_length)
data = number2base(data,2,data_length)
codeword = conv_encode_list(data,[p1,p2])

def Bernoulli(n,p):
    return([int(rand.random()<p) for n in range( n )])

def bsc(x,p):
    n = len(x)
    return([int((a+b)%2) for a,b in zip(x,Bernoulli(n,p))])

def awgn(x,sigma):
  return([a+rand.gauss(0,sigma) for a in x])



#############diagram######################################################################################################

def conv_state_diagram(octal_polynomials):
  binary_polynomials = []
  for p in octal_polynomials:
    binary_polynomials.append(base2base(p,8,2))
  output, state = conv_encode_symbol(0, binary_polynomials)
  L = len(state)
  n_nodes = 2**L
  g = nx.DiGraph()
  g.add_nodes_from(range(n_nodes))
  for in_node in range(n_nodes):
    state=number2base(in_node,2,L)
    g.nodes[in_node]["state"] = ''.join(str(n) for n in state)
    for input_symbol in [0,1]:
      state=number2base(in_node,2,L)
      output,state = conv_encode_symbol(input_symbol,binary_polynomials,state)
      out_node = base2number(state,2)
      g.add_edge(in_node,out_node,in_out=f"{str(input_symbol)}/{''.join(str(n) for n in output)}")
  return(g)

sd = conv_state_diagram([[5],[7]])
############ puncturing ##############################################################

def puncture_sequence(encoded_sequence, puncturing_pattern=[1,1,0,0,1,0,0,1]):
    punctured_sequence = []
    
    # Process the sequence in groups of 8 bits
    for i in range(0, len(encoded_sequence), len(puncturing_pattern)):
        block = encoded_sequence[i:i + len(puncturing_pattern)]  # Get a block of 8 bits
        punctured_block = [bit for bit, keep in zip(block, puncturing_pattern) if keep]  # Apply pattern
        punctured_sequence.extend(punctured_block)  # Add filtered bits to output
    
    return punctured_sequence

#############trellis######################################################################################################

def sd2trellis(sd, n_stages):
  td = nx.DiGraph()
  td.add_node((0,0),state=sd.nodes[0]["state"],stage=0,index=0) # (stage,node_no)
  for stage in range(n_stages):
    stage_nodes = [n for n in td.nodes() if n[0] == stage]
    for node in stage_nodes:
      for neighbour in sd.neighbors(node[1]):
        if not td.has_node((stage+1,neighbour)):
          td.add_node((stage+1,neighbour), state=sd.nodes[neighbour]["state"],stage=stage+1,index=neighbour)
        td.add_edge(node,(stage+1,neighbour),in_out=sd.edges[node[1],neighbour]["in_out"])
  return(td)

def draw_trellis(td, node_labels="state", edge_labels="in_out"):
  # we labeled our nodes in the trellis (stage,index) where index was their
  # label in the state diagram. Unfortunately, while networkX appears to support
  # nodes indexed as a tuple, its graphic rendering functions don't seem to
  # support it so we first need to convert the trellis to a graph whose nodes
  # are indexed as integers
  mapping = {n: idx for idx,n in enumerate(td.nodes)}
  g = nx.relabel_nodes(td,mapping)
  f = plt.figure()
  f.set_figwidth(25)
  f.set_figheight(5)
  # we now prepare a node layout where the x dimension is the trellis stage
  # and the y dimension is the node index in the state diagram
  layout = []
  for n in g.nodes:
    layout.append((g.nodes[n]["stage"],g.nodes[n]["index"]))
  # and now to draw the trellis
  elabels={edge: g.edges[edge][edge_labels] for edge in g.edges()}
  nx.draw_networkx_edge_labels(g,layout, edge_labels=elabels,font_color='red')
  nlabels={node: g.nodes[node][node_labels] for node in g.nodes()}
  nx.draw(g, layout, labels=nlabels, with_labels=True, font_weight='bold')

td=sd2trellis(sd,10)

######################### viterbi ########################################################################################

def Viterbi(y, td, channel='bsc'):

  # phase 1: compute gammas
  N = len([int(a) for a in td.edges[((0,0),(1,0))]['in_out'][2:]]) # code rate 1/N
  for edge in td.edges:
    edge_output = [int(a) for a in td.edges[edge]['in_out'][2:]]
    y_pos = td.nodes[edge[0]]['stage']*N
    y_slice = y[y_pos:y_pos+N]
    if channel == 'bsc':
      td.edges[edge]['gamma'] = sum([int(a != b) for a,b in zip(edge_output, y_slice)])
    elif channel == 'bec':
      diff = [0 if (b=='?' or a == b) else 1 for a,b in zip(edge_output, y_slice)]
      if sum(diff) == 0:
        td.edges[edge]['gamma'] = 0
      else:
        td.edges[edge]['gamma'] = 1 # 1 is infinity for the BEC!
    elif channel == 'awgn':
      modulated_output = [1-2*a for a in edge_output]
      td.edges[edge]['gamma'] = sum([(a-b)**2 for a,b in zip(modulated_output, y_slice)])

  # phase 2: compute alphas
  # This phase can be implemented in a purely "forward" way if you are confident
  # of being able to visit your vertices in an order that will only ever visit
  # a vertex when all of its incoming edges are connected to vertices whose
  # alpha have already been computed. However, for more complicated trellises,
  # it may not be evident to design a forward path that visits all nodes in
  # order and hence there is a need for an approach that will work on every
  # trellis so we can use it in all applications of the Viterbi algorithm. The
  # approach implemented here is a bit was suggested to me by my 4th year
  # project student Omar Zaman: we start from the TOOR and operate a "stack" of
  # nodes that we are unable to compute. For the node at the top of the stack,
  # we check if its incoming neighbours have already been resolved. If yes, we
  # resolve the node and remove it from the stack. If not, we add the unresolved
  # incoming neighbours to the stack.
  for node in td.nodes:
    td.nodes[node]['alpha'] = -1 # initialise all vertices to -1
  root = (0,0)
  toor = (len(y)/N, 0)
  td.nodes[root]['alpha'] = 0 # initialise root vertex to 0
  vstack = [toor]
  while vstack:
    cnode = vstack[-1]
    incoming = [e for e in td.edges if e[1]==cnode]
    sources = [e[0] for e in incoming]
    source_alphas = [td.nodes[n]['alpha'] for n in sources]
    if any([x == -1 for x in source_alphas]):  # some origins are uncomputed
      for n in sources:
        if (td.nodes[n]["alpha"] == -1):
          vstack.append(n)
    else: # all origins computed, node can be resolved
      min_alpha = -1
      winner = -1
      for e in incoming:
        td.edges[e]['winning'] = 'loser' # preset all incoming to losing
        candidate_alpha = td.edges[e]['gamma'] + td.nodes[e[0]]['alpha']
        if min_alpha == -1 or candidate_alpha < min_alpha:
          winner = e
          min_alpha = candidate_alpha
      td.nodes[cnode]['alpha'] = min_alpha
      td.edges[winner]['winning'] = 'winner' # record the winning incoming
      vstack.pop() # delete vertex from stack

  # phase 3: backtrack to read out the winning path and metric
  cnode = toor;
  decoded = []
  while (td.nodes[cnode]["stage"] > 0):
    winner = [e for e in td.edges if e[1]==cnode and td.edges[e]['winning']=="winner"]
    winner = winner[0]
    decoded.append(int(td.edges[winner]['in_out'][0]))
    cnode = winner[0]
  decoded.reverse()
  min_metric = td.nodes[toor]['alpha']

  return decoded, min_metric
######################### draft bcjr ######################################################################################

def draft_bcjr(y, td, channel='bsc', parameter=0.1):
  N = len([int(a) for a in td.edges[((0,0),(1,0))]['in_out'][2:]]) # code rate 1/N
  state_length = len(td.nodes[(0,0)]["state"])
  n_stages = int(len(y)/N)

  # phase 1: compute gammas
  for e in td.edges:
    edge_output = [int(a) for a in td.edges[e]["in_out"][2:]]
    y_pos = td.nodes[e[0]]["stage"]*N
    y_slice = y[y_pos:y_pos+N]
    if channel == 'bsc':
      hamming_distance = sum([int(a != b) for a,b in zip(edge_output, y_slice)])
      td.edges[e]["gamma"] = np.power(parameter, hamming_distance)*np.power(1-parameter,N-hamming_distance)
    elif channel == 'bec':
      count_erasures = sum([1 if a=='?' else 0 for a in y_slice])
      count_differences = sum([1 if (b != '?' and a != b) else 0 for a,b in zip(edge_output, y_slice)])
      if (count_differences > 0):
        td.edges[e]["gamma"] = 0
      else:
        td.edges[e]["gamma"] = np.power(parameter, count_erasures)*np.power(1-parameter, N-count_erasures)
    elif channel == 'awgn':
      modulated_output = np.array([1-2*a for a in edge_output])
      squared_distance = np.sum(np.power(y_slice - modulated_output,2))
      td.edges[e]["gamma"] = np.exp(-squared_distance/np.power(parameter,2.0))/np.sqrt(2.0*np.pi)/parameter

  # phase 2: run the forward backward algorithm
  for n in td.nodes:
    td.nodes[n]["alpha"] = 0 # initialise all vertices to 0
    td.nodes[n]["beta"] = 0 # initialise all vertices to 0
  root = (0,0)
  toor = (n_stages, 0)
  td.nodes[root]["alpha"] = 1 # initialise root alpha to 1
  td.nodes[toor]["beta"] = 1 # initialise toor beta to 1
  # forward
  for stage in range(1,n_stages+1):
    for n in [n for n in td.nodes if n[0]==stage]:
      for e in [e for e in td.edges if e[1] == n]:
        td.nodes[n]["alpha"] += td.edges[e]["gamma"]*td.nodes[e[0]]["alpha"]
  # backward
  for stage in range(n_stages-1,-1,-1):
    for n in [n for n in td.nodes if n[0]==stage]:
      for e in [e for e in td.edges if e[0]==n]:
        td.nodes[n]["beta"] += td.edges[e]["gamma"]*td.nodes[e[1]]["beta"]
  # now to multiply the alphas, betas and gammas
  for e in td.edges:
    td.edges[e]["gamma"] *= td.nodes[e[0]]["alpha"]*td.nodes[e[1]]["beta"]

  # Phase 3: summarise data symbol probabilities
  app = []
  decoded = []
  for stage in range(n_stages):
    sum0 = 0
    sum1 = 0
    for e in [e for e in td.edges if e[0][0]==stage]:
      if td.edges[e]["in_out"][0] == '0':
        sum0 += td.edges[e]["gamma"]
      else:
        sum1 += td.edges[e]["gamma"]
    with np.errstate(divide='ignore'):
      app.append(np.log(sum0/sum1))
    decoded.append(int(app[-1] < 0))

  return app, decoded

######################### bcjr ############################################################################################
def logmax(x,y):
  return(max(x,y)+ np.log(1+np.exp(min(x,y)-max(x,y))))

# this can be applied recursively to lists
def logmaxlist(x):
  if len(x) == 1:
        return x[0]
  if len(x) == 2:
    return(logmax(x[0],x[1]))
  else:
    return(logmax(x[0],logmaxlist(x[1:])))

def log_bcjr(llr_input, td, apriori_llr=None):

    N = len([int(a) for a in td.edges[((0,0), (1,0))]['in_out'][2:]])  # Code rate 1/N
    n_stages = len(llr_input) // N

    # Initialize trellis metrics
    for n in td.nodes:
        td.nodes[n]["alpha"] = float('-inf')  # log(0)
        td.nodes[n]["beta"] = float('-inf')

    td.nodes[(0, 0)]["alpha"] = 0  # log(1) = 0
    td.nodes[(n_stages, 0)]["beta"] = 0  # log(1) = 0

    # Phase 1: Compute gamma values using LLRs
    for e in td.edges:
        edge_output = [int(a) for a in td.edges[e]["in_out"][2:]]
        y_pos = td.nodes[e[0]]["stage"] * N
        y_slice = llr_input[y_pos:y_pos + N]

        # Compute branch metric γ = L(y) * output + a-priori LLR
        gamma = sum(y * (1 - 2 * b) for y, b in zip(y_slice, edge_output))
        if apriori_llr is not None:
            gamma += apriori_llr[td.nodes[e[0]]["stage"]]  # Add a-priori info

        td.edges[e]["gamma"] = gamma

    # Phase 2: Compute Alpha (Forward)
    root = (0,0)
    toor = (n_stages,0)
    for n in td.nodes:
      td.nodes[n]["alpha"] = 0 # initialise all vertices to 0
      td.nodes[n]["beta"] = 0 # initialise all vertices to 0
    # forward
    for stage in range(1,n_stages+1):
      for n in [n for n in td.nodes if n[0]==stage]:
        ee = [e for e in td.edges if e[1]==n]
        if (len(ee)==0):
          continue
        e = ee[0]
        td.nodes[n]["alpha"] = td.edges[e]["gamma"] + td.nodes[e[0]]["alpha"]
        ee.pop(0)
        for e in ee:
          new_alpha = td.edges[e]["gamma"]+td.nodes[e[0]]["alpha"]
          td.nodes[n]["alpha"] = logmax(td.nodes[n]["alpha"],new_alpha)
    # backward
    for stage in range(n_stages-1,-1,-1):
      for n in [n for n in td.nodes if n[0]==stage]:
        ee = [e for e in td.edges if e[0]==n]
        if (len(ee)==0):
          continue
        e = ee[0]
        td.nodes[n]["beta"] = td.edges[e]["gamma"] + td.nodes[e[1]]["beta"]
        ee.pop(0)
        for e in ee:
          new_beta = td.edges[e]["gamma"]+td.nodes[e[1]]["beta"]
          td.nodes[n]["beta"] = logmax(td.nodes[n]["beta"],new_beta)
    # now to "multiply" the alphas, betas and gammas by adding the logs
    for e in td.edges:
      td.edges[e]["gamma"] += td.nodes[e[0]]["alpha"]+td.nodes[e[1]]["beta"]

    # Phase 4: Compute APPs and Extrinsic LLRs
    app = []
    extrinsic = []
    decoded = []

    for stage in range(n_stages):
        zero_edges = [td.edges[e]["gamma"] for e in td.edges if e[0][0] == stage and td.edges[e]["in_out"][0] == '0']
        one_edges = [td.edges[e]["gamma"] for e in td.edges if e[0][0] == stage and td.edges[e]["in_out"][0] == '1']

        zero_log_max = logmaxlist(zero_edges) if zero_edges else float('-inf')
        one_log_max = logmaxlist(one_edges) if one_edges else float('-inf')

        llr = zero_log_max - one_log_max
        app.append(llr)

        # Compute extrinsic information: L_e = L_a - L(y)
        extrinsic.append(llr - (apriori_llr[stage] if apriori_llr is not None else 0))

        decoded.append(int(llr < 0))  # MAP decision

    return app, extrinsic, decoded

def generate_interleaver(n):

    return np.random.permutation(range(n)).tolist()

def interleave(data, perm):
    return [data[k] for k in perm]

def deinterleave(data, perm):
    iperm = sorted([(x, y) for x, y in zip(perm, range(len(perm)))], key=lambda x: x[0])
    iperm = [x[1] for x in iperm]
    return [data[k] for k in iperm]

data_length = 50
# conv code polynomials
p1 = [1,5]
p2 = [1,7]
data = [0]*data_length # all zero data and codeword suffice for simulation
# data = = rand.getrandbits(data_length) # commands for random data removed
# data = number2base(data,2,data_length).# test with random data if you prefer
perm = generate_interleaver(8)
codeword = conv_encode_list(data,[p1,p2])
codeword = puncture_sequence(codeword + interleave(codeword,perm))


# we rebuild the state diagram and trellis here to match the lengths
# (in case you fiddled with parameters in the earlier part of the lab)
sd = conv_state_diagram([p1,p2])
n_delays = len(sd.nodes[0]["state"])
td = sd2trellis(sd, data_length+n_delays)

max_number_of_iterations = 5
# now for the simulation
eb_n0 = [-2,-1.5,-1,0,0.5,1,1.5] # in dB
def llr_to_bit(llr):
    # Convert LLR value to hard bit decision
    return 1 if llr > 0 else 0


def turbo_decoding(received_bits, number_of_iterations):
       # Initialize variables for the decoded bits and soft information (LLR)
    decoded_bits = []
    llr_values = received_bits  # These are your soft received values (LLR)

    # Step 2: Iterative decoding (for a given number of iterations)
    for iteration in range(number_of_iterations):
        
        # Step 3: Decode using the first BCJR decoder
        ap, ext, llr_output1 = log_bcjr(llr_values, td, apriori_llr=None)  # For first decoder
        decoded_bits1 = [llr_to_bit(llr) for llr in llr_output1]

        # Step 4: Interleave and decode using the second BCJR decoder
        ap, ext, llr_output2 = log_bcjr(llr_values, td, apriori_llr=None)  # For second decoder
        decoded_bits2 = [llr_to_bit(llr) for llr in llr_output2]
        interleaved_decoded_bits2 = interleave(decoded_bits2, perm)  # Interleave the second decoder's output

        # Combine the decoded bits from both decoders (soft information is exchanged)
        llr_values = llr_output1 + llr_output2
        decoded_bits = decoded_bits1 + interleaved_decoded_bits2

    return decoded_bits

def run_error_calcs(eb_n0):
  error_rates = []

  for i in range(1,(max_number_of_iterations)):
    for e in eb_n0:
      block_errors_Viterbi = 0
      bit_errors_logBCJR = 0
      block_errors_logBCJR = 0
      blocks = 0
      sigma = np.power(10,-e/20)/np.sqrt(2)
      print(f"Eb/N0: {e}, Sigma = {sigma}")
      for k in range(2000): # max number of transmitted blocks.
        if blocks % 100 == 0:
          print(f"Blocks so far: {blocks}")
        # ideally this should be 1e5 but we don't have enough computing power....
        y = awgn([1-x*2 for x in codeword], sigma)
        dlogb = turbo_decoding(y, i)
        dv,sqd = Viterbi(y,td,"awgn")
        b = sum(dlogb)  # just add the bits to count errors!
        if b>0:
          block_errors_logBCJR +=1
        bit_errors_logBCJR += b
        b=sum(dv)
        if b>0:
          block_errors_Viterbi += 1
        blocks+=1
        if block_errors_Viterbi > 100: # you can stop when you've seen 100 errors
          break
      BlER_logBCJR = block_errors_logBCJR/blocks
      BER_logBCJR = bit_errors_logBCJR/(blocks*data_length)
      if i==1:
        error_rates.append([e,BlER_logBCJR,BER_logBCJR])
      else:
        for d in range(len(error_rates)):
          if error_rates[d][0] == e:
            error_rates[d].extend([BlER_logBCJR,BER_logBCJR])
  e = np.array(error_rates)
  return e 


e=run_error_calcs(eb_n0)
def Q(x):
  return 0.5 - 0.5*special.erf(x/np.sqrt(2))
e_saved=np.array([[-2.00000000e+00,  3.36666667e-01,  3.30666667e-02,  5.10000000e-01, 5.21333333e-02,  3.40000000e-01,  3.17333333e-02],
           [-1.50000000e+00,  2.20524017e-01,  2.08733624e-02,  3.38427948e-01, 3.06113537e-02,  2.24890830e-01,  2.10917031e-02],
           [-1.00000000e+00,  1.62903226e-01,  1.34516129e-02,  2.91935484e-01,  2.33870968e-02,  1.66129032e-01 , 1.30967742e-02],
           [ 0.00000000e+00,  3.20000000e-02,  1.99000000e-03,  1.12500000e-01, 6.30000000e-03,  3.25000000e-02,  2.02000000e-03],
           [ 5.00000000e-01,  1.55000000e-02,  1.03000000e-03,  7.10000000e-02, 3.73000000e-03,  1.55000000e-02,  9.60000000e-04],
           [ 1.00000000e+00,  6.50000000e-03,  2.50000000e-04,  4.10000000e-02, 1.60000000e-03,  6.50000000e-03,  2.50000000e-04],
           [ 1.50000000e+00,  2.00000000e-03,  1.20000000e-04,  3.00000000e-02, 1.20000000e-03,  2.00000000e-03,  1.20000000e-04]])

turbo_data = [[-2, 0.5047846889952153, 0.031626794258373205, 0.489010989010989, 0.021043956043956043, 0.45363408521303256, 0.016741854636591478, 0.4229828850855746, 0.014034229828850857], [-1.5, 0.36335403726708076, 0.019099378881987577, 0.3887884267631103, 0.01511754068716094, 0.3572593800978793, 0.012920065252854813, 0.329073482428115, 0.010095846645367413], [-1, 0.2995121951219512, 0.013326829268292683, 0.27076923076923076, 0.009353846153846154, 0.2644098810612992, 0.008453796889295517, 0.26371308016877637, 0.007974683544303798], [0, 0.142, 0.00532, 0.137, 0.00361, 0.1295, 0.00365, 0.1335, 0.00382], [0.5, 0.091, 0.00303, 0.082, 0.00221, 0.086, 0.00264, 0.087, 0.00213], [1, 0.0555, 0.00145, 0.055, 0.00126, 0.058, 0.00164, 0.053, 0.00127], [1.5, 0.041, 0.00104, 0.035, 0.00078, 0.0385, 0.00108, 0.032, 0.00077]]
ebn0 = e[:,0]
sigma = np.power(10,-ebn0/20)/np.sqrt(2)
uncoded_rate = [Q(1/x) for x in sigma]
print("runtime =", time.time()-start_time)
plt.semilogy(ebn0,uncoded_rate)
plt.semilogy(ebn0,[x[1:] for x in e]) # other error rates
plt.grid()
plt.xlabel("Eb/N0")
plt.ylabel("Error rates")
plt.legend(["Uncoded bit error rate", "Turbo Coding block error rate", "Turbo Coding bit error rate","2","2b", "3", "3b"])
plt.plot()
plt.show()
print("THE LOG BCJR BIT ERROR RATE CANNOT BE RIGHT!")
