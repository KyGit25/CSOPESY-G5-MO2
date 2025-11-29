/*
G5 - PROCESS SCHEDULER AND CLI (DESAMITO, MARISTELA, SANLUIS)
*/

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <map>
#include <deque>
#include <stack>
#include <memory>    
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <random>
#include <chrono>
#include <iomanip>      
#include <algorithm>
#include <cmath>

/**
 * @brief Defines the config params read from config.txt
 */
struct Config {
    int num_cpu = 0;
    std::string scheduler = "rr";
    int quantum_cycles = 1;
    int batch_process_freq = 0;
    int min_ins = 0;
    int max_ins = 0;
    int delays_per_exec = 0;
    size_t max_overall_mem = 0;
    size_t mem_per_frame = 0;
    size_t min_mem_per_proc = 0;
    size_t max_mem_per_proc = 0;
};

/**
 * @brief Types of instructions for the barebones interpreter.
 * FOR_START and FOR_END are used to implement the FOR loop logic.
 */
enum class InstructionType {
    PRINT,
    DECLARE,
    ADD,
    SUBTRACT,
    SLEEP,
    FOR_START,
    FOR_END,
    READ,
    WRITE
};

/**
 * @brief Represents a single barebones instruction.
 */
struct Instruction {
    InstructionType type;
    std::vector<std::string> args;
};

/**
 * @brief Represents the state of a process.
 */
enum class ProcessState {
    READY,
    RUNNING,
    SLEEPING,
    TERMINATED
};

/**
 * @brief Information needed to manage a FOR loop's execution.
 */
struct ForLoopInfo {
    int start_pc;             // The PC of the FOR_START instruction + 1
    int remaining_iterations;
};

/**
 * @brief Represents a single process in the emulator.
 */
struct Process {
    int id;
    std::string name;
    ProcessState state;
    std::vector<Instruction> instructions;
    int pc = 0; // Program Counter
    std::map<std::string, uint16_t> variables;
    std::stack<ForLoopInfo> for_loop_stack;
    std::map<int, uint16_t> memory_data;

    // Execution & Scheduling
    uint64_t wake_up_tick = 0; // Tick at which this process wakes from SLEEP
    int core_id = -1;            // CPU core it's running on (-1 if not running)
    int total_instructions = 0;
    int executed_instructions = 0;

    size_t memory_required = 0;

    Process(int id, std::string name, int total_ins, size_t mem_req)
        : id(id), name(std::move(name)), state(ProcessState::READY), 
          total_instructions(total_ins), memory_required(mem_req) {
        
        // Get and format the start time
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&in_time_t), "%m/%d/%Y %H:%M:%S");
        start_time_str = ss.str();
    }

    // Logging & Reporting
    std::string start_time_str;
    std::vector<std::string> logs;
    std::mutex log_mutex; // Protects the 'logs' vector
    
};

struct Frame {
    int frame_id;
    int process_id; // -1 if free
    int page_id;    // which page of the process
    bool dirty;     // Modified?
    std::chrono::steady_clock::time_point last_accessed; 
    std::vector<uint8_t> data; 
};

// Configuration
Config global_config;

// Emulator State
std::atomic<bool> os_running(true);
std::atomic<bool> initialized(false);
std::atomic<bool> scheduler_running(false); // For scheduler-start/stop
std::atomic<uint64_t> cpu_ticks(0);
std::atomic<int> process_id_counter(0);

// Process Lists 
std::vector<std::shared_ptr<Process>> all_processes;
std::mutex all_processes_mutex;

std::deque<std::shared_ptr<Process>> ready_queue;
std::mutex ready_queue_mutex;
std::condition_variable ready_queue_cv;

std::vector<std::shared_ptr<Process>> sleeping_processes;
std::mutex sleeping_processes_mutex;

std::vector<std::shared_ptr<Process>> finished_processes;
std::mutex finished_processes_mutex;

// CPU Core State
std::vector<std::shared_ptr<Process>> current_core_processes; // [core_id] -> Process
std::mutex current_core_processes_mutex;
// Using unique_ptr to manage atomic variables since they're not copyable/movable
std::vector<std::unique_ptr<std::atomic<bool>>> core_busy_status; // For utilization report

// Thread Management
std::vector<std::thread> cpu_core_threads;
std::thread manager_thread;

// Random Number Generator (for process generation)
std::mt19937 rng;

// ==========================================
// MEMORY MANAGER (STEP 2 ADDITION)
// ==========================================

class MemoryManager {
private:
    std::vector<Frame> ram;
    size_t total_memory;
    size_t frame_size;
    size_t num_frames;
    
    // Backing Store: Key: <ProcessID, PageID>, Value: Data
    std::map<std::pair<int, int>, std::vector<uint8_t>> backing_store;
    std::string backing_store_file = "csopesy-backing-store.txt";
    
    // Thread safety
    std::mutex mem_mutex;

    // Stats for vmstat
    size_t paged_in = 0;
    size_t paged_out = 0;

    void updateBackingStoreFile() {
        // Requirement: Backing store represented as a text file 
        std::ofstream file(backing_store_file);
        if (file.is_open()) {
            file << "Backing Store State:\n";
            for (auto const& [key, val] : backing_store) {
                file << "Process: " << key.first << " | Page: " << key.second << "\n";
            }
            file.close();
        }
    }

public:
    MemoryManager() {}

    void init(size_t max_mem, size_t frame_sz) {
        std::lock_guard<std::mutex> lock(mem_mutex);
        total_memory = max_mem;
        frame_size = frame_sz;
        // Calculate number of frames 
        num_frames = (frame_size > 0) ? total_memory / frame_size : 0;
        
        ram.clear();
        for (size_t i = 0; i < num_frames; ++i) {
            Frame f;
            f.frame_id = i;
            f.process_id = -1; // Free
            f.page_id = -1;
            f.dirty = false;
            f.data.resize(frame_size, 0);
            ram.push_back(f);
        }
        
        // Reset backing store file
        std::ofstream file(backing_store_file);
        file << "Initialized.\n";
        file.close();
    }

    // Returns true if access successful (in RAM), false if Page Fault handled (simulation delay needed)
    bool accessMemory(int pid, int page_id, bool isWrite) {
        std::lock_guard<std::mutex> lock(mem_mutex);

        // 1. Check if in RAM (TLB/Table Walk simulation)
        for (auto &f : ram) {
            if (f.process_id == pid && f.page_id == page_id) {
                f.last_accessed = std::chrono::steady_clock::now();
                if (isWrite) f.dirty = true;
                return true; // Hit
            }
        }

        // 2. Page Fault - Find Victim (LRU)
        int victim_idx = -1;
        
        // Try to find free frame first
        for(int i=0; i < num_frames; ++i) {
            if(ram[i].process_id == -1) {
                victim_idx = i;
                break;
            }
        }

        // If no free frame, LRU Eviction
        if (victim_idx == -1) {
            auto oldest = std::chrono::steady_clock::now();
            for(int i=0; i < num_frames; ++i) {
                if(ram[i].last_accessed < oldest) {
                    oldest = ram[i].last_accessed;
                    victim_idx = i;
                }
            }
        }
        
        if (victim_idx == -1) return false; // Should not happen unless 0 memory

        Frame& victim = ram[victim_idx];

        // 3. Page Out (Swap out victim if occupied)
        if (victim.process_id != -1) {
            backing_store[{victim.process_id, victim.page_id}] = victim.data;
            paged_out++;
            updateBackingStoreFile();
        }

        // 4. Page In (Swap in new page)
        victim.process_id = pid;
        victim.page_id = page_id;
        victim.last_accessed = std::chrono::steady_clock::now();
        victim.dirty = isWrite;
        
        // Retrieve from backing store if exists, else zero init
        if (backing_store.count({pid, page_id})) {
            victim.data = backing_store[{pid, page_id}];
        } else {
            std::fill(victim.data.begin(), victim.data.end(), 0);
        }
        
        paged_in++;
        return true; 
    }
    
    // For vmstat and process-smi
    void getStats(size_t &total, size_t &used, size_t &free_mem, size_t &pin, size_t &pout) {
        std::lock_guard<std::mutex> lock(mem_mutex);
        total = total_memory;
        used = 0;
        for(const auto& f : ram) {
            if(f.process_id != -1) used += frame_size;
        }
        free_mem = total - used;
        pin = paged_in;
        pout = paged_out;
    }
    
    size_t getFrameSize() { return frame_size; }
};

// Instantiate the Global Memory Manager
MemoryManager mem_manager;

void main_cli_loop();
bool load_config(const std::string& filename);
std::shared_ptr<Process> generate_dummy_process();
void cpu_worker_function(int core_id);
void manager_thread_function();
void execute_instruction(const std::shared_ptr<Process>& process, const Instruction& instr, bool& jumped, bool& process_slept);
void display_report(bool write_to_file);
void enter_screen_mode(const std::shared_ptr<Process>& process);
std::vector<std::string> split_string(const std::string& s, char delimiter);
int64_t parse_value(const std::shared_ptr<Process>& process, const std::string& arg);
std::string get_current_timestamp_log();

/**
 * @brief Splits a string by a delimiter.
 */
std::vector<std::string> split_string(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

/**
 * @brief Gets a timestamp string for process logs.
 */
std::string get_current_timestamp_log() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    // Format: (MM/DD/YYYY HH:MM:SS)
    ss << "(" << std::put_time(std::localtime(&in_time_t), "%m/%d/%Y %H:%M:%S") << ")";
    return ss.str();
}

/**
 * @brief Parses an argument, which could be a literal number or a variable name.
 * @return The value of the argument. Returns 0 if variable not found.
 */
int64_t parse_value(const std::shared_ptr<Process>& process, const std::string& arg) {
    if (arg.empty()) return 0;

    // Check if it's a variable
    if (arg[0] == '$') {
        std::string var_name = arg.substr(1);
        if (process->variables.count(var_name)) {
            return process->variables[var_name];
        } else {
            return 0;
        }
    }

    // Otherwise, it's a literal
    try {
        return std::stoll(arg);
    } catch (...) {
        return 0; // Failed to parse
    }
}

/**
 * @brief Clamps a value to the uint16_t range (0 to 65535).
 */
uint16_t clamp_uint16(int64_t value) {
    if (value < 0) return 0;
    if (value > 65535) return 65535; // UINT16_MAX
    return static_cast<uint16_t>(value);
}

/**
 * @brief Generates a new dummy process with a random set of instructions.
 */
std::shared_ptr<Process> generate_dummy_process() {
    // 1. Calculate Memory Requirement (Power of 2 check)
    // We want a power of 2 between min_mem_per_proc and max_mem_per_proc
    std::vector<size_t> valid_mem_sizes;
    size_t curr = 1;
    while (curr < global_config.min_mem_per_proc) curr <<= 1;
    while (curr <= global_config.max_mem_per_proc) {
        valid_mem_sizes.push_back(curr);
        curr <<= 1;
    }
    
    // Fallback if config is weird
    if (valid_mem_sizes.empty()) valid_mem_sizes.push_back(global_config.min_mem_per_proc);

    std::uniform_int_distribution<int> dist_mem_idx(0, valid_mem_sizes.size() - 1);
    size_t mem_req = valid_mem_sizes[dist_mem_idx(rng)];

    // 2. Create Process with Memory Param
    std::uniform_int_distribution<int> dist_ins(global_config.min_ins, global_config.max_ins);
    int num_instructions = dist_ins(rng);
    int current_id = process_id_counter.fetch_add(1);
    
    std::stringstream ss_name;
    ss_name << "p" << std::setfill('0') << std::setw(3) << current_id;
    std::string process_name = ss_name.str();

    auto process = std::make_shared<Process>(current_id, process_name, num_instructions, mem_req);

    // 3. Generate Random Instructions
    // Increased range to 7 to include READ/WRITE
    std::uniform_int_distribution<int> dist_op(0, 7); 
    std::uniform_int_distribution<int> dist_val(1, 100);
    std::uniform_int_distribution<int> dist_var(0, 2); 
    std::vector<std::string> var_names = {"x", "y", "z"};

    int nest_level = 0;
    const int max_nest_level = 3;

    for (int i = 0; i < num_instructions; ++i) {
        Instruction instr;
        int op = dist_op(rng);

        // Force loop close if near end
        if (nest_level > 0 && i >= num_instructions - nest_level) {
            op = 6; 
        }

        switch (op) {
            case 0: // PRINT
                instr.type = InstructionType::PRINT;
                instr.args.push_back("\"Hello from " + process_name + "!\"");
                break;
            case 1: // DECLARE
                instr.type = InstructionType::DECLARE;
                instr.args.push_back("$" + var_names[dist_var(rng)]);
                instr.args.push_back(std::to_string(dist_val(rng)));
                break;
            case 2: // ADD
                instr.type = InstructionType::ADD;
                instr.args.push_back("$" + var_names[dist_var(rng)]);
                instr.args.push_back("$" + var_names[dist_var(rng)]);
                instr.args.push_back("$" + var_names[dist_var(rng)]);
                break;
            case 3: // SUBTRACT
                instr.type = InstructionType::SUBTRACT;
                instr.args.push_back("$" + var_names[dist_var(rng)]);
                instr.args.push_back("$" + var_names[dist_var(rng)]);
                instr.args.push_back("$" + var_names[dist_var(rng)]);
                break;
            case 4: // SLEEP
                instr.type = InstructionType::SLEEP;
                instr.args.push_back(std::to_string(dist_val(rng) % 10 + 1));
                break;
            case 5: // FOR_START
                if (nest_level < max_nest_level) {
                    instr.type = InstructionType::FOR_START;
                    instr.args.push_back(std::to_string(dist_val(rng) % 5 + 1));
                    process->instructions.push_back(instr);
                    nest_level++;
                } else {
                    instr.type = InstructionType::PRINT;
                    instr.args.push_back("\"Max nest reached!\"");
                }
                break;
            case 6: // FOR_END
                if (nest_level > 0) {
                    instr.type = InstructionType::FOR_END;
                    process->instructions.push_back(instr);
                    nest_level--;
                } else {
                    instr.type = InstructionType::PRINT;
                    instr.args.push_back("\"No loop to end!\"");
                }
                break;
            
            // --- NEW INSTRUCTIONS ---
            case 7: // READ (READ $var addr)
            case 8: // WRITE (WRITE addr value)
                {
                    // Generate random address within process memory limit
                    // Align to 2 bytes (word alignment) just to be safe/realistic
                    size_t max_addr = mem_req - 2; 
                    if (max_addr <= 0) max_addr = 0;
                    
                    std::uniform_int_distribution<size_t> dist_addr(0, max_addr);
                    size_t addr = dist_addr(rng);
                    
                    // Convert to hex string "0x..."
                    std::stringstream ss_addr;
                    ss_addr << "0x" << std::hex << addr;
                    
                    if (op == 7) { // READ
                        instr.type = InstructionType::READ;
                        instr.args.push_back("$" + var_names[dist_var(rng)]); // dest var
                        instr.args.push_back(ss_addr.str()); // address
                    } else { // WRITE
                        instr.type = InstructionType::WRITE;
                        instr.args.push_back(ss_addr.str()); // address
                        instr.args.push_back(std::to_string(dist_val(rng))); // value
                    }
                }
                break;
        }

        if (op < 5 || (op > 6)) { // Don't add if handled inside case 5/6 logic
             if(instr.type != InstructionType::FOR_START && instr.type != InstructionType::FOR_END) {
                 process->instructions.push_back(instr);
             }
        }
    }

    // Close remaining loops
    while (nest_level > 0) {
        Instruction instr;
        instr.type = InstructionType::FOR_END;
        process->instructions.push_back(instr);
        nest_level--;
    }
    
    process->total_instructions = process->instructions.size();
    return process;
}

/**
 * @brief Executes a single barebones instruction for a given process.
 *
 * @param process The process executing the instruction.
 * @param instr The instruction to execute.
 * @param jumped Output param: set to true if the PC was modified (i.e., a FOR loop)
 * @param process_slept Output param: set to true if the instruction was SLEEP
 */
void execute_instruction(
    const std::shared_ptr<Process>& process,
    const Instruction& instr,
    bool& jumped,
    bool& process_slept) 
{
    jumped = false;
    process_slept = false;

    try {
        switch (instr.type) {
            case InstructionType::PRINT: {
                // Arg0: The message string, e.g., "$var" or "\"literal msg\""
                std::string msg = instr.args[0];
                std::string output;
                
                if (msg.front() == '$') {
                    // Print variable
                    output = std::to_string(parse_value(process, msg));
                } else {
                    // Print literal string
                    output = msg;
                    if (output.front() == '"' && output.back() == '"') {
                        output = output.substr(1, output.length() - 2); // Remove quotes
                    }
                    
                    // Replace *$process_name* with the actual name
                    std::string placeholder = "*$process_name*";
                    size_t pos = output.find(placeholder);
                    if (pos != std::string::npos) {
                        output.replace(pos, placeholder.length(), process->name);
                    }
                }

                std::string log_entry = get_current_timestamp_log() + " Core:" + 
                                        std::to_string(process->core_id) + " " + output;
                
                std::lock_guard<std::mutex> lock(process->log_mutex);
                process->logs.push_back(log_entry);
                break;
            }

            case InstructionType::DECLARE: {
                // Arg0: var_name (e.g., "$x"), Arg1: value (e.g., "100" or "$y")
                std::string var_name = instr.args[0].substr(1); // Remove '$'
                uint16_t value = clamp_uint16(parse_value(process, instr.args[1]));
                process->variables[var_name] = value;
                break;
            }

            case InstructionType::ADD: {
                // Arg0: var1, Arg1: var2, Arg2: result_var
                int64_t val1 = parse_value(process, instr.args[0]);
                int64_t val2 = parse_value(process, instr.args[1]);
                std::string result_var = instr.args[2].substr(1); // Remove '$'
                process->variables[result_var] = clamp_uint16(val1 + val2);
                break;
            }

            case InstructionType::SUBTRACT: {
                // Arg0: var1, Arg1: var2, Arg2: result_var
                int64_t val1 = parse_value(process, instr.args[0]);
                int64_t val2 = parse_value(process, instr.args[1]);
                std::string result_var = instr.args[2].substr(1); // Remove '$'
                process->variables[result_var] = clamp_uint16(val1 - val2);
                break;
            }

            case InstructionType::SLEEP: {
                // Arg0: ticks
                int64_t sleep_ticks = parse_value(process, instr.args[0]);
                process->state = ProcessState::SLEEPING;
                // Add current_tick + sleep_ticks.
                // We use +1 because the current tick is already in progress.
                process->wake_up_tick = cpu_ticks.load() + sleep_ticks + 1;
                process_slept = true;
                break;
            }

            case InstructionType::FOR_START: {
                // Arg0: repeats
                int iterations = static_cast<int>(parse_value(process, instr.args[0]));
                // Push the *next* PC, which is the start of the loop body
                process->for_loop_stack.push({process->pc + 1, iterations});
                break;
            }

            case InstructionType::FOR_END: {
                if (process->for_loop_stack.empty()) {
                    // Malformed code, just continue
                    break;
                }
                
                ForLoopInfo& loop = process->for_loop_stack.top();
                loop.remaining_iterations--;

                if (loop.remaining_iterations > 0) {
                    // Jump back to the start of the loop body
                    process->pc = loop.start_pc;
                    jumped = true;
                } else {
                    // Loop finished, pop it
                    process->for_loop_stack.pop();
                }
                break;
            }

            case InstructionType::READ: {
                // Format: READ $var 0xAddress
                // Arg0: $var, Arg1: Address (hex)
                std::string var_name = instr.args[0].substr(1); // Remove $
                std::string addr_str = instr.args[1];
                int addr = std::stoi(addr_str, nullptr, 16); // Parse Hex
                
                // 1. Get value from process memory (or 0 if not init)
                uint16_t val = 0;
                if (process->memory_data.count(addr)) {
                    val = process->memory_data[addr];
                }
                
                // 2. Store into variable
                process->variables[var_name] = val;
                
                // Log
                {
                    std::lock_guard<std::mutex> lock(process->log_mutex);
                    std::stringstream ss;
                    ss << get_current_timestamp_log() << " Core:" << process->core_id 
                       << " READ " << var_name << " (" << val << ") from " << addr_str;
                    process->logs.push_back(ss.str());
                }
                break;
            }

            case InstructionType::WRITE: {
                // Format: WRITE 0xAddress Value
                // Arg0: Address (hex), Arg1: Value (or $var)
                std::string addr_str = instr.args[0];
                int addr = std::stoi(addr_str, nullptr, 16); // Parse Hex
                
                int64_t val_raw = parse_value(process, instr.args[1]);
                uint16_t val = clamp_uint16(val_raw);
                
                // 1. Write to process memory
                process->memory_data[addr] = val;
                
                // Log
                {
                    std::lock_guard<std::mutex> lock(process->log_mutex);
                    std::stringstream ss;
                    ss << get_current_timestamp_log() << " Core:" << process->core_id 
                       << " WROTE " << val << " to " << addr_str;
                    process->logs.push_back(ss.str());
                }
                break;
            }
        }
    } catch (const std::exception& e) {
        // Handle potential errors, e.g., out-of-bounds args
        std::string log_entry = get_current_timestamp_log() + " Core:" + 
                                std::to_string(process->core_id) + " ERROR executing instruction: " + e.what();
        std::lock_guard<std::mutex> lock(process->log_mutex);
        process->logs.push_back(log_entry);
    }
}

/**
 * @brief The function executed by each CPU Core thread.
 *
 * This function loops, waiting for processes on the ready_queue.
 * When a process is available, it "executes" it for a time slice
 * (if RR) or until completion/sleep (if FCFS).
 */
void cpu_worker_function(int core_id) {
    while (os_running) {
        std::shared_ptr<Process> process;

        // 1. Get a process from the ready queue
        {
            std::unique_lock<std::mutex> lock(ready_queue_mutex);
            // Wait until the queue is not empty OR the OS is shutting down
            ready_queue_cv.wait(lock, [&] {
                return !ready_queue.empty() || !os_running;
            });

            if (!os_running) {
                return; // OS is shutting down
            }

            // Get the process
            if (global_config.scheduler == "fcfs") {
                process = ready_queue.front();
                ready_queue.pop_front();
            } else { // "rr"
                process = ready_queue.front();
                ready_queue.pop_front();
            }
            
            // Mark this core as busy and running the process
            core_busy_status[core_id]->store(true);
            process->state = ProcessState::RUNNING;
            process->core_id = core_id;

            {
                std::lock_guard<std::mutex> ccp_lock(current_core_processes_mutex);
                current_core_processes[core_id] = process;
            }
        } // Release ready_queue_mutex

        // 2. Execute the process
        bool process_finished = false;
        bool process_slept = false;
        
        // FCFS runs until block/terminate, RR runs for a quantum
        int cycles_to_run = (global_config.scheduler == "rr")
                                ? global_config.quantum_cycles
                                : process->total_instructions; // Effectively "run to completion"

        for (int i = 0; i < cycles_to_run; ++i) {
            if (process->pc >= process->instructions.size()) {
                process_finished = true;
                break;
            }

            const Instruction& instr = process->instructions[process->pc];

            // --- MEMORY MANAGEMENT CHECK ---
            int page_to_access = -1;
            bool is_write = false;
            
            if (instr.type == InstructionType::DECLARE) {
                // Symbol table is usually at the start of memory (Page 0)
                page_to_access = 0;
                is_write = true;
            } 
            else if (instr.type == InstructionType::READ) {
                // Check Address in Arg1
                try {
                    int addr = std::stoi(instr.args[1], nullptr, 16);
                    page_to_access = addr / global_config.mem_per_frame;
                } catch(...) {}
            } 
            else if (instr.type == InstructionType::WRITE) {
                // Check Address in Arg0
                try {
                    int addr = std::stoi(instr.args[0], nullptr, 16);
                    page_to_access = addr / global_config.mem_per_frame;
                    is_write = true;
                } catch(...) {}
            }

            // If this instruction touches memory, check the Memory Manager
            if (page_to_access != -1) {
                // This simulates the Demand Paging delay
                // If it returns false (conceptually), it handled a fault.
                // Since our accessMemory implementation handles the swap synchronously, 
                // we just call it to ensure stats (paged in/out) are updated.
                mem_manager.accessMemory(process->id, page_to_access, is_write);
                
                // OPTIONAL: Add a small sleep to simulate disk I/O if a Page Fault occurred
                // For now, we rely on the busy-wait loop below.
            }
            // -------------------------------

            bool jumped = false;
            execute_instruction(process, instr, jumped, process_slept);
            
            process->executed_instructions++;

            if (!jumped) {
                process->pc++;
            }

            // Simulate the instruction delay as a busy-wait
            if (global_config.delays_per_exec > 0) {
                uint64_t start_tick = cpu_ticks.load();
                uint64_t end_tick = start_tick + global_config.delays_per_exec;
                while (cpu_ticks.load() < end_tick && os_running) {
                    std::this_thread::yield(); 
                }
            } else {
                // "If zero, each instruction is executed per CPU cycle."
                uint64_t current_tick = cpu_ticks.load();
                while (cpu_ticks.load() == current_tick && os_running) {
                     std::this_thread::yield();
                }
            }

            if (process_slept || !os_running) {
                break; // Instruction was SLEEP or OS is stopping
            }
        }

        // 3. Post-execution: Update process state and lists
        {
            // Mark core as idle
            std::lock_guard<std::mutex> ccp_lock(current_core_processes_mutex);
            current_core_processes[core_id] = nullptr;
        }
        process->core_id = -1;
        core_busy_status[core_id]->store(false);


        if (!os_running) {
            // If OS is stopping, put process back in ready queue
            process->state = ProcessState::READY;
            std::lock_guard<std::mutex> lock(ready_queue_mutex);
            ready_queue.push_front(process); // Put back for clean shutdown
            return;
        }

        if (process_finished || process->pc >= process->instructions.size()) {
            process->state = ProcessState::TERMINATED;
            std::lock_guard<std::mutex> lock(finished_processes_mutex);
            finished_processes.push_back(process);
        } else if (process_slept) {
            // State was already set to SLEEPING
            std::lock_guard<std::mutex> lock(sleeping_processes_mutex);
            sleeping_processes.push_back(process);
        } else {
            // Quantum expired (for RR)
            process->state = ProcessState::READY;
            std::lock_guard<std::mutex> lock(ready_queue_mutex);
            ready_queue.push_back(process); // Put at the back
        }
    }
}

/**
 * @brief The function for the single manager thread.
 *
 * This thread is the heartbeat of the OS. It:
 * 1. Increments cpu_ticks.
 * 2. Wakes up processes from the sleeping_processes list.
 * 3. Generates new batch processes if scheduler_running is true.
 */
void manager_thread_function() {
    uint64_t next_generation_tick = 0;
    
    if (global_config.batch_process_freq > 0) {
        next_generation_tick = global_config.batch_process_freq;
    } else {
        // Avoid divide-by-zero, just set it far in the future
        next_generation_tick = (uint64_t)-1;
    }

    while (os_running) {
        // 1. Sleep for 1ms to simulate one tick passing
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        
        // 2. Increment the global CPU tick
        uint64_t current_tick = cpu_ticks.fetch_add(1) + 1;

        // 3. Check sleeping processes
        {
            std::lock_guard<std::mutex> lock(sleeping_processes_mutex);
            // Iterate and erase idiom
            sleeping_processes.erase(
                std::remove_if(sleeping_processes.begin(), sleeping_processes.end(),
                    [&](const std::shared_ptr<Process>& p) {
                        if (p->wake_up_tick <= current_tick) {
                            p->state = ProcessState::READY;
                            std::lock_guard<std::mutex> ready_lock(ready_queue_mutex);
                            ready_queue.push_back(p);
                            return true; // Remove from sleeping list
                        }
                        return false; // Keep sleeping
                    }),
                sleeping_processes.end());
        }

        // 4. Check if we need to generate new processes
        if (scheduler_running && current_tick >= next_generation_tick) {
            if(global_config.batch_process_freq > 0) {
                 // Generate a new process
                std::shared_ptr<Process> new_process = generate_dummy_process();

                // Add to all_processes list
                {
                    std::lock_guard<std::mutex> lock(all_processes_mutex);
                    all_processes.push_back(new_process);
                }
                // Add to ready queue
                {
                    std::lock_guard<std::mutex> lock(ready_queue_mutex);
                    ready_queue.push_back(new_process);
                }
                
                // Set the next generation time
                next_generation_tick = current_tick + global_config.batch_process_freq;
            }
        }
        
        // 5. Notify any waiting CPU cores
        ready_queue_cv.notify_all();
    }
}

/**
 * @brief Prints the report for 'screen -ls' or 'report-util'.
 */
void display_report(bool write_to_file) {
    std::stringstream ss;

    // 1. Calculate CPU Utilization
    int busy_cores = 0;
    for (size_t i = 0; i < global_config.num_cpu; ++i) {
        if (core_busy_status[i]->load()) {
            busy_cores++;
        }
    }
    double utilization = (global_config.num_cpu > 0)
                           ? (static_cast<double>(busy_cores) / global_config.num_cpu) * 100.0
                           : 0.0;

    ss << "CPU utilization: " << std::fixed << std::setprecision(0) << utilization << "%\n";
    ss << "Cores used: " << busy_cores << "\n";
    ss << "Cores available: " << global_config.num_cpu - busy_cores << "\n\n";

    // Lock all process lists needed for the report
    std::lock_guard<std::mutex> all_lock(all_processes_mutex);
    std::lock_guard<std::mutex> finished_lock(finished_processes_mutex);
    std::lock_guard<std::mutex> ccp_lock(current_core_processes_mutex);

    // 2. Running Processes
    ss << "Running processes:\n";
    for (int i = 0; i < global_config.num_cpu; ++i) {
        auto process = current_core_processes[i];
        if (process) {
            ss << "  " << process->name << "  "
               << "(" << process->start_time_str << ")  Core: " << i << "   "
               << process->executed_instructions << " / " << process->total_instructions << "\n";
        }
    }

    // 3. Finished Processes
    ss << "\nFinished processes:\n";
    for (const auto& process : finished_processes) {
        ss << "  " << process->name << "  "
           << "(" << process->start_time_str << ")  Finished  "
           << process->executed_instructions << " / " << process->total_instructions << "\n";
    }

    // 4. Output
    std::cout << ss.str();
    if (write_to_file) {
        std::ofstream out_file("csopesy-log.txt");
        out_file << ss.str();
        out_file.close();
        std::cout << "\nReport generated at ./csopesy-log.txt\n";
    }
}

/**
 * @brief Enters the interactive "screen" for a specific process.
 */
void enter_screen_mode(const std::shared_ptr<Process>& process) {
    if (!process) {
        std::cout << "Error: Process not found.\n";
        return;
    }

    std::string line;
    while (os_running) {
        // Clear screen (crude, but functional)
        std::cout << "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n";
        
        std::cout << "--- Process Screen: " << process->name << " ---\n";
        std::cout << "ID:           " << process->id << "\n";
        std::cout << "Status:       ";
        switch (process->state) {
            case ProcessState::READY: std::cout << "Ready\n"; break;
            case ProcessState::RUNNING: std::cout << "Running (Core " << process->core_id << ")\n"; break;
            case ProcessState::SLEEPING: std::cout << "Sleeping (wake at " << process->wake_up_tick << ")\n"; break;
            case ProcessState::TERMINATED: std::cout << "Terminated\n"; break;
        }
        std::cout << "Instructions: " << process->executed_instructions << " / " << process->total_instructions << "\n";
        std::cout << "PC:           " << process->pc << "\n";
        
        std::cout << "\n--- Variables ---\n";
        if (process->variables.empty()) {
            std::cout << "(No variables declared)\n";
        }
        for (const auto& pair : process->variables) {
            std::cout << "$" << pair.first << " = " << pair.second << "\n";
        }

        std::cout << "\n--- Logs ---\n";
        {
            std::lock_guard<std::mutex> lock(process->log_mutex);
            if (process->logs.empty()) {
                std::cout << "(No log output)\n";
            }
            // Print only the last 15 logs for readability
            int start = std::max(0, static_cast<int>(process->logs.size()) - 15);
            for (size_t i = start; i < process->logs.size(); ++i) {
                std::cout << process->logs[i] << "\n";
            }
        }
        
        if (process->state == ProcessState::TERMINATED) {
            std::cout << "\n--- FINISHED ---";
        }

        std::cout << "\n\nType 'exit' to return to main menu: ";
        
        // This is a simple blocking poll.
        if (!std::getline(std::cin, line)) {
            break; // EOF
        }
        if (line == "exit") {
            break;
        }
    }
}

/**
 * @brief Finds a process by name (e.g., "p001")
 */
std::shared_ptr<Process> find_process_by_name(const std::string& name) {
    std::lock_guard<std::mutex> lock(all_processes_mutex);
    auto it = std::find_if(all_processes.begin(), all_processes.end(), 
        [&name](const std::shared_ptr<Process>& p) {
        return p->name == name;
    });

    if (it != all_processes.end()) {
        return *it;
    }
    
    // Check finished processes too
    std::lock_guard<std::mutex> finished_lock(finished_processes_mutex);
    it = std::find_if(finished_processes.begin(), finished_processes.end(), 
        [&name](const std::shared_ptr<Process>& p) {
        return p->name == name;
    });
    
    if (it != finished_processes.end()) {
        return *it;
    }

    return nullptr;
}


/**
 * @brief Loads configuration from "config.txt".
 */
bool load_config(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << "\n";
        return false;
    }

    std::string line, key, value;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        if (!(iss >> key >> value)) { continue; }

        try {
            if (key == "num-cpu") global_config.num_cpu = std::stoi(value);
            else if (key == "scheduler") global_config.scheduler = value;
            else if (key == "quantum-cycles") global_config.quantum_cycles = std::stoi(value);
            else if (key == "batch-process-freq") global_config.batch_process_freq = std::stoi(value);
            else if (key == "min-ins") global_config.min_ins = std::stoi(value);
            else if (key == "max-ins") global_config.max_ins = std::stoi(value);
            else if (key == "delays-per-exec") global_config.delays_per_exec = std::stoi(value);
            else if (key == "max-overall-mem") global_config.max_overall_mem = std::stoul(value);
            else if (key == "mem-per-frame") global_config.mem_per_frame = std::stoul(value);
            else if (key == "min-mem-per-proc") global_config.min_mem_per_proc = std::stoul(value);
            else if (key == "max-mem-per-proc") global_config.max_mem_per_proc = std::stoul(value);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing config line: " << line << "\n";
            return false;
        }
    }

    // Validate config
    if (global_config.num_cpu <= 0) {
        std::cerr << "Error: num-cpu must be > 0.\n"; return false;
    }
    
    // Initialize RNG
    unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
    rng.seed(seed);
    
    return true;
}

void cmd_vmstat() {
    size_t total, used, free_mem, pin, pout;
    mem_manager.getStats(total, used, free_mem, pin, pout);
    
    std::cout << "\n";
    std::cout << std::left << std::setw(15) << "Total Mem" << std::setw(15) << "Used Mem" << std::setw(15) << "Free Mem" << "\n";
    std::cout << std::left << std::setw(15) << total << std::setw(15) << used << std::setw(15) << free_mem << "\n";
    std::cout << "\n";
    std::cout << std::left << std::setw(15) << "Idle Ticks" << std::setw(15) << "Active Ticks" << std::setw(15) << "Total Ticks" << "\n";
    // Calculate stats roughly based on ticks
    uint64_t total_ticks = cpu_ticks.load();
    // In this simple sim, active is roughly total - idle. 
    // We don't track global idle perfectly, so we'll approximate for display:
    std::cout << std::left << std::setw(15) << "N/A" << std::setw(15) << total_ticks << std::setw(15) << total_ticks << "\n";
    std::cout << "\n";
    std::cout << std::left << std::setw(15) << "Pages In" << std::setw(15) << "Pages Out" << "\n";
    std::cout << std::left << std::setw(15) << pin << std::setw(15) << pout << "\n";
}

void cmd_process_smi() {
    size_t total, used, free_mem, pin, pout;
    mem_manager.getStats(total, used, free_mem, pin, pout);
    
    double util = (total > 0) ? (double)used / total * 100.0 : 0.0;
    
    std::cout << "\n------------------------------------------\n";
    std::cout << "| PROCESS-SMI V01.00 Driver Version: 01.00 |\n";
    std::cout << "------------------------------------------\n";
    std::cout << "CPU-Util: 100%\n"; // Sim is always running
    std::cout << "Memory Usage: " << used << "B / " << total << "B\n";
    std::cout << "Memory Util: " << std::fixed << std::setprecision(1) << util << "%\n";
    std::cout << "------------------------------------------\n";
    std::cout << "Running processes and memory usage:\n";
    std::cout << "------------------------------------------\n";
    
    std::lock_guard<std::mutex> lock(all_processes_mutex);
    for (const auto& p : all_processes) {
        if (p->state != ProcessState::TERMINATED) {
            std::cout << std::left << std::setw(15) << p->name << " " << p->memory_required << "B\n";
        }
    }
    std::cout << "------------------------------------------\n";
}

std::shared_ptr<Process> create_process_from_script(std::string name, size_t mem, std::string script) {
    int current_id = process_id_counter.fetch_add(1);
    
    // Create process (instructions empty at first)
    auto process = std::make_shared<Process>(current_id, name, 0, mem);
    
    std::vector<std::string> lines = split_string(script, ';');
    for (std::string& line : lines) {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string opcode;
        ss >> opcode;
        
        Instruction instr;
        if (opcode == "DECLARE") {
            instr.type = InstructionType::DECLARE;
            std::string var, val; ss >> var >> val;
            instr.args = {var, val};
        } else if (opcode == "PRINT") {
            instr.type = InstructionType::PRINT;
            // Extract everything inside quotes
            size_t first = line.find('"');
            size_t last = line.find_last_of('"');
            if (first != std::string::npos && last != std::string::npos) {
                instr.args = {line.substr(first, last - first + 1)};
            }
        } else if (opcode == "ADD") {
            instr.type = InstructionType::ADD;
            std::string v1, v2, v3; ss >> v1 >> v2 >> v3;
            instr.args = {v1, v2, v3};
        } else if (opcode == "SUBTRACT") {
            instr.type = InstructionType::SUBTRACT;
            std::string v1, v2, v3; ss >> v1 >> v2 >> v3;
            instr.args = {v1, v2, v3}; // v1 = v2 - v3 ?? Standard typically dest last
        } else if (opcode == "READ") {
            instr.type = InstructionType::READ;
            std::string var, addr; ss >> var >> addr;
            instr.args = {var, addr};
        } else if (opcode == "WRITE") {
            instr.type = InstructionType::WRITE;
            std::string addr, val; ss >> addr >> val;
            instr.args = {addr, val};
        } else if (opcode == "SLEEP") {
            instr.type = InstructionType::SLEEP;
            std::string val; ss >> val;
            instr.args = {val};
        }
        
        process->instructions.push_back(instr);
    }
    
    process->total_instructions = process->instructions.size();
    return process;
}

int main() {
    std::cout << "\n----------------------------------------\n";
    std::cout << "Welcome to CSOPESY Emulator!\n";
    std::cout << "\nDevelopers:\n";
    std::cout << " - Desamito, Hector Francis Seigmund\n";
    std::cout << " - Maristela, Kyle Gabriel\n";
    std::cout << " - San Luis, Owen Philip\n";
    std::cout << "\nLast Updated: " << __DATE__ << "\n";
    std::cout << "----------------------------------------\n";

    std::string line;
    while (os_running) {
        std::cout << "\nroot:\\> ";
        if (!std::getline(std::cin, line)) {
            if (os_running) {
                line = "exit"; // Handle Ctrl+D or EOF
            } else {
                break;
            }
        }

        std::vector<std::string> tokens = split_string(line, ' ');
        if (tokens.empty()) continue;

        std::string command = tokens[0];

        if (command == "initialize") {
            if (initialized) {
                std::cout << "Emulator already initialized.\n";
                continue;
            }

            if (load_config("config.txt")) {
                std::cout << "Configuration loaded. Starting " << global_config.num_cpu << " cores.\n";
                
                mem_manager.init(global_config.max_overall_mem, global_config.mem_per_frame);

                // Resize core state vectors
                current_core_processes.resize(global_config.num_cpu);
                core_busy_status.resize(global_config.num_cpu);
                for (int i = 0; i < global_config.num_cpu; ++i) {
                    core_busy_status[i] = std::make_unique<std::atomic<bool>>(false);
                }

                // Start CPU core threads
                for (int i = 0; i < global_config.num_cpu; ++i) {
                    cpu_core_threads.emplace_back(cpu_worker_function, i);
                }
                
                // Start the manager thread
                manager_thread = std::thread(manager_thread_function);

                initialized = true;
                std::cout << "Emulator initialized.\n";
            } else {
                std::cout << "Failed to initialize. Check config.txt.\n";
            }
            continue;
        }

            if (!initialized) {
                if (command == "exit") {
                    os_running = false;
                    ready_queue_cv.notify_all(); // Wake any waiting threads
                    std::cout << "Emulator terminating." << std::endl;
                    break;
                }

                std::cout << "Error: You must run 'initialize' first." << std::endl;
                continue;
            }

        if (command == "exit") {
            os_running = false;
            ready_queue_cv.notify_all(); // Wake all threads
            
            // Join all threads
            manager_thread.join();
            for (auto& t : cpu_core_threads) {
                t.join();
            }
            
            std::cout << "Emulator terminating.\n";
            break;

        } else if (command == "scheduler-start") {
            scheduler_running = true;
            std::cout << "Batch process generator started.\n";

        } else if (command == "scheduler-stop") {
            scheduler_running = false;
            std::cout << "Batch process generator stopped.\n";

        } else if (command == "report-util") {
            display_report(true);

        } else if (command == "vmstat") {
            if (initialized) cmd_vmstat();
            else std::cout << "Run initialize first.\n";

        } else if (command == "process-smi") {
            if (initialized) cmd_process_smi();
            else std::cout << "Run initialize first.\n";

        } else if (command == "screen") {
             // Handle screen variants
            if (tokens.size() > 1 && tokens[1] == "-ls") {
                display_report(false);
            } 
            else if (tokens.size() > 2 && tokens[1] == "-r") {
                // screen -r <name>
                auto p = find_process_by_name(tokens[2]);
                if (p) enter_screen_mode(p);
                else std::cout << "Process not found.\n";
            }
            else if (tokens.size() > 2 && tokens[1] == "-s") {
                // screen -s <name> (Optional: <mem>)
                std::string name = tokens[2];
                // Default mem if not provided or parse
                // NOTE: generate_dummy_process calculates its own mem, 
                // but if user provides it in -s, we might want to override.
                // For now, keep simple:
                auto p = generate_dummy_process(); 
                p->name = name; // Override name
                
                // Register
                {
                    std::lock_guard<std::mutex> lock(all_processes_mutex);
                    all_processes.push_back(p);
                }
                {
                    std::lock_guard<std::mutex> lock(ready_queue_mutex);
                    ready_queue.push_back(p);
                }
                std::cout << "Process " << name << " created.\n";
                enter_screen_mode(p);
            }
            else if (tokens.size() >= 4 && tokens[1] == "-c") {
                // screen -c <name> <mem> "<instructions>"
                std::string name = tokens[2];
                size_t mem = std::stoul(tokens[3]);
                
                // Reconstruct the script part (it might have spaces)
                size_t first_quote = line.find('"');
                size_t last_quote = line.find_last_of('"');
                
                if (first_quote != std::string::npos && last_quote != std::string::npos) {
                    std::string script = line.substr(first_quote + 1, last_quote - first_quote - 1);
                    auto p = create_process_from_script(name, mem, script);
                    
                    // Register
                    {
                        std::lock_guard<std::mutex> lock(all_processes_mutex);
                        all_processes.push_back(p);
                    }
                    {
                        std::lock_guard<std::mutex> lock(ready_queue_mutex);
                        ready_queue.push_back(p);
                    }
                    std::cout << "Custom process " << name << " created.\n";
                    enter_screen_mode(p);
                } else {
                    std::cout << "Invalid command format. Missing quotes?\n";
                }
            }
            else {
                std::cout << "Invalid screen command.\n";
            }
        }
        else {
            std::cout << "Unknown command.\n";
        }
    }

    return 0;
}

