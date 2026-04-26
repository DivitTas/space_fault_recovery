Space Fault Recovery Agent
                            
  The problem
                                                                                        
  When we send probes to deep space, we're bottlenecked by the speed of light. A radio  
  signal to Mars takes around half an hour each way. For time-sensitive failures, that  
  round-trip means hours can pass before engineers on Earth even see a problem, let     
  alone fix it. The further out we go, the worse this gets — communication with a
  Voyager-style probe can take a full day round-trip.

  Deep space exploration will always be capped as long as probes rely on humans to      
  recover from failures. A probe that can diagnose and repair itself opens up missions
  we currently can't attempt.                                                           
                  
  Why RL?

  Most spacecraft today recover from faults using rule-based systems — long lists of "if
   this sensor reads X, do Y" hardcoded by engineers. These work beautifully for
  situations the engineers anticipated. They fall apart when something happens that     
  nobody put on the list.

  JAXA's Hayabusa mission is a good example. A solar flare damaged its solar panels,    
  which cascaded into malfunctioning reaction wheels (the spinning gyros that point the
  probe). The mission was eventually recovered, but only after engineers spent countless
   hours on Earth crafting a stabilization plan that wasn't in the original playbook.

  A reinforcement learning agent doesn't need every scenario hardcoded ahead of time.   
  Instead, it learns from experience how to recognize trouble and recover — including
  from situations its designers never imagined.                                         
                  
  The environment

  We built a simulated spacecraft with 7 classes of faults spanning power, attitude,    
  comms, and thermal systems. Crucially, the systems are interdependent: a power fault
  starves the comms transmitter; a damaged solar panel forces the battery to drain      
  faster; a thermal anomaly can corrupt the star tracker the probe uses to point itself.

  The agent doesn't get a clean readout of what's actually broken. It sees only what its
   sensors report, and sensors themselves can be degraded or lying. To find out the
  truth, the agent has to actively investigate — running one of 17 diagnostic and repair
   commands to query a subsystem, cross-check sensors, shed load, or attempt a repair.
  Every diagnostic costs time, and every wrong repair makes things worse.

  This is what makes the problem genuinely hard: the agent isn't just picking the right 
  action from a clear menu. It has to figure out what's wrong before it can fix it, with
   incomplete and sometimes misleading information.                                     
                  
  What has the agent learnt?

  We trained a small (1.5B-parameter) language model with GRPO — a reinforcement        
  learning method that lets the agent try different repair strategies, score them by how
   well the spacecraft recovers, and gradually shift toward the strategies that work.   
                  
  After [N] training episodes on an A100 GPU, the agent's mission success rate moved    
  from [X]% before training to [Y]% after — a [Z]% improvement on a held-out set of
  fault scenarios it had never seen.                                                    
                  
  What does that look like in practice? A few behaviors stood out:                      
  
  - Diagnose before acting. Early in training the agent would jump straight to repairs  
  and often pick the wrong one. By the end of training it learned to first run a few
  diagnostic queries — checking battery, solar panels, and attitude — before committing 
  to a fix.       
  - [Behavior 2 — fill in once you've watched a few rollouts. e.g., "Power before 
  pointing": realised that fixing power has to happen before stabilizing attitude,      
  because pointing maneuvers drain the battery.]
  - [Behavior 3 — fill in. e.g., "Knew when to give up on a subsystem": if science      
  instruments were drawing too much power during a crisis, the agent learned to shed    
  those loads to save the probe rather than try to keep everything running.]
                                                                                        
  It's not perfect. The agent still fails on [the hardest fault combinations / cascading
   scenarios / specific class]. But the result we cared about — can a small model learn 
  to recover a spacecraft from injected faults without ever being told the rules? — is  
  starting to look like yes.

  What's next

  This is hackathon-stage work, not a flight-ready system. The next steps are: more     
  training, harder fault scenarios, and eventually testing whether what the agent
  learned in our simulator transfers to real spacecraft simulations like NASA's GMAT.   
                  
  The bigger picture: probes that can fix themselves aren't a far-future fantasy. The   
  pieces — small efficient models, RL training that runs on a single GPU, simulators
  rich enough to teach real strategies — exist today. The interesting question isn't    
  whether this is possible. It's how soon we put it in space.

  I genuinely think that a concept like this can change how we perform deep space exploration. Perhaps, one day, we'll have probes that don't suffer the same fate as, for instance, opportunity on distant lands where communication with humanity takes very long. 
