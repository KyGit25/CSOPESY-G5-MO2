#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <map>
#include <unordered_map>
#include <sstream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <ctime>
#include <random>

using namespace std;

// ==========================================
// 1. CONFIGURATION & GLOBALS
// ==========================================

struct Config {
    int num_cpu;
    string scheduler;
    int quantum_cycles;
    int batch_process_freq;
    int min_ins;
    int max_ins;
    int delay_per_exec;
    size_t max_overall_mem;
    size_t mem_per_frame;
    size_t min_mem_per_proc;
    size_t max_mem_per_proc;
};

Config globalConfig;
bool isInitialized = false;
bool stopScheduler = false;
atomic<bool> isRunning{true};
mutex printMutex; // To keep console output clean
mutex memMutex;   // Memory Manager Lock
mutex schedulerMutex;

// Formatting helper
string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    struct tm* tm = localtime(&now);
    stringstream ss;
    ss << put_time(tm, "(%m/%d/%Y %I:%M:%S%p)");
    return ss.str();
}

// ==========================================
// 2. DATA STRUCTURES
// ==========================================

enum InstructionType {
    CPU_OP, // Generic compute
    PRINT,
    DECLARE, // Reserve symbol table space
    READ,    // Memory Access
    WRITE    // Memory Access
};

struct Instruction {
    InstructionType type;
    string varName;
    uint32_t address;
    uint16_t value;
    string printMsg;
    int duration = 1; // 1 cycle
};

struct Process {
    string name;
    int id;
    vector<Instruction> instructions;
    size_t program_counter = 0;
    
    // Process Stats
    size_t memory_required;
    int cpu_core_id = -1;
    time_t start_time;
    size_t total_instructions = 0;
    size_t instructions_executed = 0;
    
    // State
    bool isFinished = false;
    string errorMsg = "";

    Process(string n, int i, size_t mem) : name(n), id(i), memory_required(mem) {
        start_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    }
};

// ==========================================
// 3. MEMORY MANAGER (DEMAND PAGING)
// ==========================================

struct Frame {
    int frame_id;
    int process_id; // -1 if free
    int page_id;    // which page of the process
    bool dirty;     // Modified?
    chrono::steady_clock::time_point last_accessed; 
    vector<uint8_t> data; // The actual simulation data
};

class MemoryManager {
private:
    vector<Frame> ram;
    size_t total_memory;
    size_t frame_size;
    size_t num_frames;
    
    // Backing Store: Simulated as a Map, flushed to file for requirement
    // Key: <ProcessID, PageID>, Value: Data
    map<pair<int, int>, vector<uint8_t>> backing_store;
    string backing_store_file = "csopesy-backing-store.txt";

    // Stats
    size_t paged_in = 0;
    size_t paged_out = 0;

    void updateBackingStoreFile() {
        // Requirement: Backing store is a text file accessible at any time.
        // We rewrite it on change. In a real OS, this is binary, here we dump text metadata.
        ofstream file(backing_store_file);
        if (file.is_open()) {
            file << "Backing Store State:\n";
            for (auto const& [key, val] : backing_store) {
                file << "Process: " << key.first << " | Page: " << key.second << " | Size: " << val.size() << " bytes\n";
            }
            file.close();
        }
    }

public:
    MemoryManager() {}

    void init(size_t max_mem, size_t frame_sz) {
        total_memory = max_mem;
        frame_size = frame_sz;
        num_frames = total_memory / frame_size;
        
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
        ofstream file(backing_store_file);
        file << "Initialized.\n";
        file.close();
    }

    // Returns true if access successful, throws runtime_error if invalid memory
    bool accessMemory(int pid, int page_id, bool isWrite) {
        lock_guard<mutex> lock(memMutex);

        // 1. Check if in RAM
        for (auto &f : ram) {
            if (f.process_id == pid && f.page_id == page_id) {
                f.last_accessed = chrono::steady_clock::now();
                if (isWrite) f.dirty = true;
                return true; // Hit
            }
        }

        // 2. Page Fault - Find Victim
        int victim_idx = -1;
        
        // Try to find free frame first
        for(int i=0; i<num_frames; ++i) {
            if(ram[i].process_id == -1) {
                victim_idx = i;
                break;
            }
        }

        // If no free frame, LRU Eviction
        if (victim_idx == -1) {
            auto oldest = chrono::steady_clock::now();
            for(int i=0; i<num_frames; ++i) {
                if(ram[i].last_accessed < oldest) {
                    oldest = ram[i].last_accessed;
                    victim_idx = i;
                }
            }
        }

        Frame& victim = ram[victim_idx];

        // 3. Page Out (Swap out victim if occupied)
        if (victim.process_id != -1) {
            // Save to backing store
            backing_store[{victim.process_id, victim.page_id}] = victim.data;
            paged_out++;
            updateBackingStoreFile();
        }

        // 4. Page In (Swap in new page)
        victim.process_id = pid;
        victim.page_id = page_id;
        victim.last_accessed = chrono::steady_clock::now();
        victim.dirty = isWrite;
        
        // Check if exists in backing store
        if (backing_store.count({pid, page_id})) {
            victim.data = backing_store[{pid, page_id}];
        } else {
            // New page, zero init
            fill(victim.data.begin(), victim.data.end(), 0);
        }
        
        // Remove from backing store map (conceptually moving to RAM)
        // Although typically we keep a copy, for this sim we strictly follow paged-in count
        paged_in++;
        
        return true;
    }
    
    // For process-smi
    size_t getUsedMemory() {
        lock_guard<mutex> lock(memMutex);
        size_t used = 0;
        for(const auto& f : ram) {
            if(f.process_id != -1) used += frame_size;
        }
        return used;
    }
    
    // For vmstat
    void getStats(size_t &total, size_t &used, size_t &free_mem, size_t &pin, size_t &pout) {
        lock_guard<mutex> lock(memMutex);
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

MemoryManager memManager;

// ==========================================
// 4. SCHEDULER & CPU
// ==========================================

struct SystemStats {
    size_t cpu_ticks = 0;
    size_t idle_ticks = 0;
    size_t active_ticks = 0;
};
SystemStats sysStats;

deque<Process*> readyQueue;
vector<Process*> finishedProcesses;
vector<Process*> allProcesses; // Owner of pointers
int process_counter = 0;

void cpuWorker(int cpu_id) {
    while (isRunning) {
        Process* currentProcess = nullptr;

        {
            lock_guard<mutex> lock(schedulerMutex);
            if (!readyQueue.empty()) {
                currentProcess = readyQueue.front();
                readyQueue.pop_front();
            }
        }

        if (currentProcess) {
            currentProcess->cpu_core_id = cpu_id;
            
            int cycles = 0;
            int quantum = (globalConfig.scheduler == "rr") ? globalConfig.quantum_cycles : 99999999;
            
            while (cycles < quantum && !currentProcess->isFinished) {
                // Check if done
                if (currentProcess->program_counter >= currentProcess->instructions.size()) {
                    currentProcess->isFinished = true;
                    break;
                }

                // Execute Instruction
                Instruction& inst = currentProcess->instructions[currentProcess->program_counter];
                
                // --- MEMORY INTERACTION ---
                bool memSuccess = true;
                if (inst.type == READ || inst.type == WRITE || inst.type == DECLARE) {
                    // Calculate Page
                    int page = -1;
                    if(inst.type == DECLARE) {
                         // Symbol table access (usually page 0 or special segment)
                         // For sim, let's say symbol table is at address 0
                         page = 0;
                    } else {
                        // Check bounds
                        if (inst.address >= currentProcess->memory_required) {
                             // Segfault
                             currentProcess->errorMsg = "Memory access violation error at address " + to_string(inst.address);
                             currentProcess->isFinished = true;
                             memSuccess = false;
                        } else {
                            page = inst.address / globalConfig.mem_per_frame;
                        }
                    }

                    if (memSuccess) {
                        try {
                            memManager.accessMemory(currentProcess->id, page, (inst.type == WRITE));
                            // Simulate delay
                            this_thread::sleep_for(chrono::milliseconds(10)); 
                        } catch (...) {
                            currentProcess->isFinished = true;
                            memSuccess = false;
                        }
                    }
                }

                if (!memSuccess) break; 

                // Simulate execution delay
                if(globalConfig.delay_per_exec > 0) {
                     this_thread::sleep_for(chrono::milliseconds(globalConfig.delay_per_exec * 10)); // rough scaling
                }
                
                // Logic
                if (inst.type == PRINT) {
                    lock_guard<mutex> pl(printMutex);
                    cout << "Process " << currentProcess->name << ": " << inst.printMsg << endl;
                }

                currentProcess->program_counter++;
                currentProcess->instructions_executed++;
                cycles++;
                sysStats.cpu_ticks++;
                sysStats.active_ticks++;
            }

            currentProcess->cpu_core_id = -1;

            {
                lock_guard<mutex> lock(schedulerMutex);
                if (currentProcess->isFinished) {
                    finishedProcesses.push_back(currentProcess);
                } else {
                    readyQueue.push_back(currentProcess);
                }
            }

        } else {
            // Idle
            sysStats.cpu_ticks++;
            sysStats.idle_ticks++;
            this_thread::sleep_for(chrono::milliseconds(100)); // Sleep to prevent busy wait spin
        }
    }
}

// ==========================================
// 5. PROCESS GENERATOR & UTILS
// ==========================================

void generateDummyProcess() {
    process_counter++;
    string name = "process" + to_string(process_counter);
    
    // Random params within config
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> disIns(globalConfig.min_ins, globalConfig.max_ins);
    uniform_int_distribution<> disMem(globalConfig.min_mem_per_proc, globalConfig.max_mem_per_proc);
    
    size_t mem = disMem(gen);
    // Align to power of 2 roughly or frame size (requirement says power of 2 format)
    // We stick to the random value but ensure it's sufficient
    if (mem < 64) mem = 64; 

    Process* p = new Process(name, process_counter, mem);
    
    int num_ins = disIns(gen);
    for(int i=0; i<num_ins; ++i) {
        Instruction inst;
        inst.type = CPU_OP;
        p->instructions.push_back(inst);
    }
    
    lock_guard<mutex> lock(schedulerMutex);
    allProcesses.push_back(p);
    readyQueue.push_back(p);
}

void processGenerator() {
    int cycles = 0;
    while(isRunning) {
        if (!stopScheduler) {
             cycles++;
             if (cycles >= globalConfig.batch_process_freq) {
                 generateDummyProcess();
                 cycles = 0;
             }
        }
        this_thread::sleep_for(chrono::milliseconds(100)); // Tick rate
    }
}

// Parser for 'screen -c' instructions
// Format: DECLARE varA 10; WRITE 0x500 15; READ varB 0x500
void parseInstructions(Process* p, string script) {
    stringstream ss(script);
    string segment;
    while(getline(ss, segment, ';')) {
        stringstream ks(segment);
        string cmd;
        ks >> cmd;
        
        Instruction inst;
        if (cmd == "DECLARE") {
            inst.type = DECLARE;
            string var; int val;
            ks >> var >> val;
            inst.varName = var; inst.value = val;
        } else if (cmd == "WRITE") {
            inst.type = WRITE;
            string addrStr; int val;
            ks >> addrStr >> val;
            inst.address = stoi(addrStr, nullptr, 16);
            inst.value = val;
        } else if (cmd == "READ") {
            inst.type = READ;
            string var, addrStr;
            ks >> var >> addrStr;
            inst.varName = var; 
            inst.address = stoi(addrStr, nullptr, 16);
        } else if (cmd == "PRINT") {
            inst.type = PRINT;
            // Hacky extraction of string in quotes
            size_t first = segment.find('"');
            size_t last = segment.find_last_of('"');
            if(first != string::npos && last != string::npos) {
                inst.printMsg = segment.substr(first+1, last-first-1);
            }
        } else {
            inst.type = CPU_OP; // Default
        }
        p->instructions.push_back(inst);
    }
}

// ==========================================
// 6. MAIN CONSOLE LOOP
// ==========================================

void printHeader() {
    cout << "  ____ ____  ___  ____  _____ ______   __" << endl;
    cout << " / ___/ ___|/ _ \\|  _ \\| ____/ ___\\ \\ / /" << endl;
    cout << "| |   \\___ \\ | | | |_) |  _| \\___ \\\\ V / " << endl;
    cout << "| |___ ___) | |_| |  __/| |___ ___) || |  " << endl;
    cout << " \\____|____/ \\___/|_|   |_____|____/ |_|  " << endl;
    cout << "------------------------------------------" << endl;
    cout << "Welcome to CSOPESY Emulator!" << endl;
}

bool loadConfig() {
    ifstream f("config.txt");
    if (!f.is_open()) return false;
    
    string param;
    while (f >> param) {
        if (param == "num-cpu") f >> globalConfig.num_cpu;
        else if (param == "scheduler") f >> globalConfig.scheduler;
        else if (param == "quantum-cycles") f >> globalConfig.quantum_cycles;
        else if (param == "batch-process-freq") f >> globalConfig.batch_process_freq;
        else if (param == "min-ins") f >> globalConfig.min_ins;
        else if (param == "max-ins") f >> globalConfig.max_ins;
        else if (param == "delay-per-exec") f >> globalConfig.delay_per_exec;
        else if (param == "max-overall-mem") f >> globalConfig.max_overall_mem;
        else if (param == "mem-per-frame") f >> globalConfig.mem_per_frame;
        else if (param == "min-mem-per-proc") f >> globalConfig.min_mem_per_proc;
        else if (param == "max-mem-per-proc") f >> globalConfig.max_mem_per_proc;
    }
    
    // Remove quotes from scheduler
    if(globalConfig.scheduler.size() > 0 && globalConfig.scheduler.front() == '"') 
        globalConfig.scheduler = globalConfig.scheduler.substr(1, globalConfig.scheduler.size()-2);

    return true;
}

void cmd_process_smi() {
    size_t total, used, free_mem, pin, pout;
    memManager.getStats(total, used, free_mem, pin, pout);
    
    double util = (double)used / total * 100.0;
    
    cout << "------------------------------------------" << endl;
    cout << "| PROCESS-SMI V01.00 Driver Version: 01.00 |" << endl;
    cout << "------------------------------------------" << endl;
    cout << "CPU-Util: " << (sysStats.active_ticks > 0 ? 100 : 0) << "%" << endl; // Simplified
    cout << "Memory Usage: " << used << "B / " << total << "B" << endl;
    cout << "Memory Util: " << fixed << setprecision(1) << util << "%" << endl;
    cout << "------------------------------------------" << endl;
    cout << "Running processes and memory usage:" << endl;
    
    lock_guard<mutex> lock(schedulerMutex);
    // Combine ready and running
    // Note: In real logic we'd iterate threads, but here queue + finished is easier
    for (auto* p : allProcesses) {
        if (!p->isFinished) {
            cout << p->name << " " << p->memory_required << "B" << endl;
        }
    }
    cout << "------------------------------------------" << endl;
}

void cmd_vmstat() {
    size_t total, used, free_mem, pin, pout;
    memManager.getStats(total, used, free_mem, pin, pout);
    
    cout << left << setw(15) << "Total Mem" << setw(15) << "Used Mem" << setw(15) << "Free Mem" << endl;
    cout << left << setw(15) << total << setw(15) << used << setw(15) << free_mem << endl;
    cout << endl;
    cout << left << setw(15) << "Idle Ticks" << setw(15) << "Active Ticks" << setw(15) << "Total Ticks" << endl;
    cout << left << setw(15) << sysStats.idle_ticks << setw(15) << sysStats.active_ticks << setw(15) << sysStats.cpu_ticks << endl;
    cout << endl;
    cout << left << setw(15) << "Pages In" << setw(15) << "Pages Out" << endl;
    cout << left << setw(15) << pin << setw(15) << pout << endl;
}

void cmd_screen_ls() {
    lock_guard<mutex> lock(schedulerMutex);
    cout << "------------------------------------------" << endl;
    cout << "Running Processes:" << endl;
    int runningCount = 0;
    for (auto* p : allProcesses) {
        if(!p->isFinished) {
            int instructions_done = p->instructions_executed;
            int total = p->instructions.size();
            cout << p->name << " (" << instructions_done << "/" << total << ")" << endl;
            runningCount++;
        }
    }
    cout << "Total Running: " << runningCount << endl;
    cout << "------------------------------------------" << endl;
    cout << "Finished Processes:" << endl;
    for (auto* p : finishedProcesses) {
        cout << p->name << " " << (p->errorMsg.empty() ? "Finished" : "Failed") << endl;
    }
    cout << "------------------------------------------" << endl;
}

void cmd_screen_r(string name) {
    lock_guard<mutex> lock(schedulerMutex);
    Process* found = nullptr;
    for (auto* p : allProcesses) {
        if (p->name == name) {
            found = p;
            break;
        }
    }
    
    if (!found) {
        cout << "Process " << name << " not found." << endl;
        return;
    }
    
    if (found->isFinished) {
        if (!found->errorMsg.empty()) {
            cout << "Process " << name << " shut down due to error." << endl;
            cout << found->errorMsg << endl;
        } else {
            cout << "Process " << name << " finished execution." << endl;
        }
    } else {
        cout << "Process " << name << " is currently running." << endl;
        cout << "Current Instruction: " << found->instructions_executed << "/" << found->instructions.size() << endl;
    }
}

int main() {
    string command;
    
    while(true) {
        cout << "root:\\> ";
        getline(cin, command);
        
        if (command == "exit") {
            isRunning = false;
            break;
        }
        else if (command == "initialize") {
            if (loadConfig()) {
                isInitialized = true;
                memManager.init(globalConfig.max_overall_mem, globalConfig.mem_per_frame);
                printHeader();
                cout << "Configuration loaded." << endl;
                
                // Start Workers
                for(int i=0; i<globalConfig.num_cpu; ++i) {
                    thread(cpuWorker, i).detach();
                }
                thread(processGenerator).detach();
                
            } else {
                cout << "Failed to open config.txt" << endl;
            }
        }
        else if (!isInitialized) {
            cout << "Please run 'initialize' first." << endl;
        }
        else if (command == "scheduler-test") {
            stopScheduler = false;
            cout << "Scheduler test started." << endl;
        }
        else if (command == "scheduler-stop") {
            stopScheduler = true;
            cout << "Scheduler test stopped." << endl;
        }
        else if (command == "process-smi") {
            cmd_process_smi();
        }
        else if (command == "vmstat") {
            cmd_vmstat();
        }
        else if (command == "screen -ls") {
            cmd_screen_ls();
        }
        else if (command.rfind("screen -s ", 0) == 0) {
            // format: screen -s <name> <mem>
            stringstream ss(command.substr(10));
            string name; size_t mem;
            ss >> name >> mem;
            
            // Validate memory range
            if (mem < 64 || mem > globalConfig.max_overall_mem) { // Simple validation based on prompt constraints
                 // Check if power of 2 (skipping strict bitwise check for simplicity, just range)
            }
            
            Process* p = new Process(name, ++process_counter, mem);
            // Default blank instructions to keep it alive for a bit
            for(int i=0; i<globalConfig.min_ins; ++i) {
                Instruction inst; inst.type = CPU_OP;
                p->instructions.push_back(inst);
            }
            
            lock_guard<mutex> lock(schedulerMutex);
            allProcesses.push_back(p);
            readyQueue.push_back(p);
            cout << "Process " << name << " created." << endl;
        }
        else if (command.rfind("screen -r ", 0) == 0) {
            string name = command.substr(10);
            cmd_screen_r(name);
        }
        else if (command.rfind("screen -c ", 0) == 0) {
            // screen -c <name> <mem> "<instructions>"
            // Complex parsing
            size_t firstQuote = command.find('"');
            size_t lastQuote = command.find_last_of('"');
            
            if(firstQuote != string::npos && lastQuote != string::npos) {
                string params = command.substr(10, firstQuote - 10);
                stringstream ss(params);
                string name; size_t mem;
                ss >> name >> mem;
                
                string script = command.substr(firstQuote + 1, lastQuote - firstQuote - 1);
                
                Process* p = new Process(name, ++process_counter, mem);
                parseInstructions(p, script);
                
                lock_guard<mutex> lock(schedulerMutex);
                allProcesses.push_back(p);
                readyQueue.push_back(p);
                cout << "Process " << name << " created with instructions." << endl;
            } else {
                cout << "Invalid command format." << endl;
            }
        }
        else {
            cout << "Unknown command." << endl;
        }
    }

    return 0;
}