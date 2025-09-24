"""Q-learning Agent class."""

from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy


class QLAgent:
    """Q-learning Agent class."""

    def __init__(self, starting_state, state_space, action_space, alpha=0.5, gamma=0.95, exploration_strategy=EpsilonGreedy()):
        """Initialize Q-learning agent."""
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.action = None
        self.alpha = alpha
        self.gamma = gamma
        #self.q_table = {self.state: [0 for _ in range(action_space.n)]}
        self.q_table = {self.state: {}}
        for phase in range(action_space.nvec[0]):  # Iterate over number of phases
            for green_time in range(action_space.nvec[1]):  # Iterate over green time options
                self.q_table[self.state][(phase, green_time)] = 0

        self.exploration = exploration_strategy
        self.acc_reward = 0
    '''
    def act(self):
        print(self.action_space)
        """Choose action based on Q-table."""
        self.action = self.exploration.choose(self.q_table, self.state, self.action_space)
        return self.action
    '''
    def act(self):
        """Choose action based on Q-table."""
        if self.state not in self.q_table:
            # Ensure the state exists in Q-table
            self.q_table[self.state] = {
                (phase, green_time): 0 
                for phase in range(self.action_space.nvec[0]) 
                for green_time in range(self.action_space.nvec[1])
            }
        
        # Select action from Q-table (exploration or exploitation)
        self.action = self.exploration.choose(self.q_table, self.state, self.action_space)
        
        # Ensure it returns a valid tuple (phase, green_time)
        if isinstance(self.action, tuple):
            return self.action
        else:
            return (0, 10)  # Default fallback if something goes wrong
    '''
    def learn(self, next_state, reward, done=False):
        """Update Q-table with new experience."""
        if next_state not in self.q_table:
            self.q_table[next_state] = [0 for _ in range(self.action_space.n)]

        s = self.state
        s1 = next_state
        a = self.action
        self.q_table[s][a] = self.q_table[s][a] + self.alpha * (
            reward + self.gamma * max(self.q_table[s1]) - self.q_table[s][a]
        )
        self.state = s1
        self.acc_reward += reward
        '''
    def learn(self, next_state, reward, done=False):
        """Update Q-table with new experience."""
        if next_state not in self.q_table:
            self.q_table[next_state] = {}  # Initialize new state with an empty dictionary
            for phase in range(self.action_space.nvec[0]):  # Iterate over possible phases
                for green_time in range(self.action_space.nvec[1]):  # Iterate over green times
                    self.q_table[next_state][(phase, green_time)] = 0  # Initialize Q-value

        s = self.state
        s1 = next_state
        a = self.action  # Action is now a tuple (phase, green_time)

        # Q-learning update rule
        self.q_table[s][a] = self.q_table[s][a] + self.alpha * (
            reward + self.gamma * max(self.q_table[s1].values()) - self.q_table[s][a]
        )

        self.state = s1
        self.acc_reward += reward

