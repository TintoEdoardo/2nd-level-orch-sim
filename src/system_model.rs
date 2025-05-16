///----------------------///
///   The System Model   ///
///----------------------///

#[derive(Eq, PartialEq, Clone)]
enum EnergySupply {
    Green,
    Regular,
}

#[derive(PartialEq, Clone)]
struct SpatialCharacteristics {
    geo_location : (f32, f32),
    energy_suppy : EnergySupply,
}

#[derive(Eq, PartialEq, Clone)]
struct ResourceProvisions {
    number_of_cpus : u8,
    memory_kb      : u32,
    frequency_mhz  : u32,
    has_gpu        : bool,
}

#[derive(Clone)]
struct Computation {
    index                   : usize,
    spatial_characteristics : SpatialCharacteristics,
    standard_execution_time : u32,
}

struct Application {
    computations     : Vec<Computation>,
    
    // The sole requirements for this experimentation. 
    number_of_server : u8,
}

/// For now, we assume that each node has but one SS. 
#[derive(Clone)]
struct Node {
    index                   : usize,
    spatial_characteristics : SpatialCharacteristics,
    resources               : ResourceProvisions,
    backlog                 : Vec<Computation>,
    sigma                   : f32,
    
    // For MCS systems. 
    // is_high                 : bool
}

struct Infrastructure {
    nodes : Vec<Node>,
}

impl Infrastructure {
    pub fn construct_federation(&self, application: &Application) -> Vec<usize> {
        
        let mut federation : Vec<usize> = Vec::new();  
        
        // Generate a federation from self.nodes. 
        // The decision is taken according to infrastructure-level criteria. 
        // In this case, we look for the node with higher frequency. 
        /* federation = self.nodes.iter().map()
            
            .clone()
            .sort_by(
                |n1, n2| n1.resources.frequency_mhz.cmp(&n2.resources.frequency_mhz));
        
        federation = self.nodes.iter().filter(
            |&node| if node.spatial_characteristics.energy_suppy == EnergySupply::Green {true} 
                            else { false })
            .collect(); */
        
        return federation
        
    }
    
}