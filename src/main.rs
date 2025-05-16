///----------------------///
///  The ADMM solution   ///
///----------------------///

mod system_model;

use std::iter::Sum;
use std::usize;
use csv::Writer;
use plotters::prelude::*;
use rand;

struct Variables {
    // x[i], with 'i' a node index.
    x: Vec<f32>
}

struct Globals {
    // z[i], with 'i' a node index.
    z: Vec<f32>
}

struct DualsVariables {
    // u[i], with 'i' a node index.
    u: Vec<f32>
}

struct  ADMMNode {
    local     : f32,
    dual      : f32,
    global    : f32,
    penalty   : f32,
    // Cost factor of computation 'c' on the current node. 
    cost_factor: f32,
}

impl ADMMNode {

    /// Local x-update.
    pub fn local_x_update(&mut self) {

        // First check if this node might host the incoming request.
        // Meaning, check whether the constraints are met with local x = 1.
        // TODO

        fn to_minimize(local:       f32,
                       dual:        f32,
                       global:      f32,
                       cost_factor: f32,
                       penalty:     f32)
                       -> f32 {
            cost_factor * local + (penalty / 2f32) * (local - global + dual).powf(2f32)
        }

        let local_at_0 =
            to_minimize(0f32, self.dual, self.global, self.cost_factor, self.penalty);
        let local_at_1 =
            to_minimize(1f32, self.dual, self.global, self.cost_factor, self.penalty);
        
        self.local = if f32::min(local_at_0, local_at_1) == local_at_0 {
            0f32
        } else {
            1f32
        };
        
    }

    /// Dual variables update (u-updates).
    pub fn local_dual_update(&mut self) {
        self.dual = self.dual + (self.local - self.global);
    }

}

/// The GlobalSolver performs the global updates.
struct GlobalSolver {
    /// x = { x_i } in the model.
    variables : Variables,

    /// z = { z_i } in the model.
    globals   : Variables,

    /// u = { u_i } in the model.
    duals     : DualsVariables,
}

impl GlobalSolver {

    /// Indicator function (delta in the model),
    /// if local is not binary, returns a high value (tending to infinity),
    /// otherwise, returns 0.
    fn indicator_function(local: f32) -> u32 {
        if local == 0f32 || local == 1f32 {
            0
        }
        else {
            // Arbitrary high (-> inf) output.
            1_000_000_000
        }
    }

    /// Update the global variable z.
    pub fn global_z_updater(&mut self) {

        // Compute the vector v.
        let mut v : Vec<f32> = Vec::new();
        for i in 0..self.variables.x.len() {
            v.push(self.variables.x[i] + self.duals.u[i]);
        }

        // Produce the subtrahend in the z-update.
        let subt = (1f32 / v.len() as f32) * (v.iter().sum::<f32>() - 1f32);

        // Update the global variables.
        for i in 0..self.globals.x.len() {
            self.globals.x[i] = v[i] - subt;
        }
    }

}


fn main() {

    let tolerance = 0.05;

    // Cost factors (c in the model).
    // In this simulation, we consider the cost factors as composed
    // exclusively by the distance 'd(\omega, n)'.
    // The algorithm to compute the cost factor corresponding to each
    // node is the following, given:
    // - A node 'n' requesting the migration,
    // - A federation 'F' associated with the application,
    // - Assuming there is a minimal physical distance between
    //   the current node 'n' and its closest node 'm' in 'F'
    // For each node in 'F' aside from 'n' and 'm' we assign a random
    // distance in the range [d(n,m)..100m]. As a result, node density
    // increases with the number of nodes.
    // The initial 'd(n,m)' is computed as '100 / #nodes'.
    fn cost_factors(max_distance: f32, number_of_nodes: u32) -> Vec<f32> {
        let mut result = Vec::new();
        let distance_n_m = max_distance / number_of_nodes as f32;
        for _i in 0..number_of_nodes {
            result.push(rand::random_range(distance_n_m..100.0));
        }
        result
    }

    // A report of the simulation outcomes.
    let mut writer =
        Writer::from_path("report.csv").expect("Could not open report.csv");

    // Run in several configurations.
    let mut nodes_range: Vec<usize> = Vec::new();
    for i in 3..99 {
        nodes_range.push(2 + i);
    }
    let penalty_range: Vec<f32> =
        vec![20f32, 30f32, 40f32, 50f32, 60f32, 70f32];

    // Vectors used for plotting the data:
    // - iterations_to_converge[\rho][#nodes]: the average number of iterations to converge,
    // 'iterations_to_converge(\rho, #nodes)' is a function referred to as 'mu(\rho, #nodes)'
    // producing the average number of iterations required to converge.
    let mut iterations_to_converge : Vec<Vec<(f32, f32)>> = Vec::new();
    /* let mut results_iter: Vec<Vec<Vec<(f32, f32)>>> = Vec::new();
    let mut results_conv: Vec<Vec<Vec<bool>>> = Vec::new();
    let mut results_iter_non_conv: Vec<Vec<(f32,f32)>> = Vec::new(); */

    // Initialize 'iterations_to_converge'.
    for _i in 0..penalty_range.len() {
        iterations_to_converge.push(Vec::new());
    }

    // Perform the experiment, repeating each configuration
    // for 'number_of_samples' times.
    let number_of_samples = 1000;
    for penalty in 0..penalty_range.len() {
        for number_of_nodes in 0..nodes_range.len() {

            // The number of samples acquired for the current
            // configuration '\rho, #nodes'.
            let mut samples : Vec::<f32> = Vec::new();
            for _sample in 0..number_of_samples {

                // The ADMM execution.
                let variables = Variables { x: vec![0f32; number_of_nodes] };
                let globals   = Variables { x: vec![1f32 / number_of_nodes as f32; number_of_nodes] };
                let duals     = DualsVariables { u: vec![0f32; number_of_nodes] };
                let cost_factors =
                    cost_factors(100f32, nodes_range[number_of_nodes] as u32);

                // Initialize the nodes in the system.
                let mut nodes: Vec<ADMMNode> = Vec::new();
                for i in 0..number_of_nodes {
                    nodes.push(ADMMNode {
                        local:       variables.x[i],
                        dual:        duals.u[i],
                        global:      globals.x[i],
                        penalty:     penalty_range[penalty],
                        cost_factor: cost_factors[i],
                    })
                }

                // And the global solver.
                let mut solver = GlobalSolver {
                    variables,
                    globals,
                    duals,
                };

                // Then simulate a consensus problem resolution.
                let mut iteration = 0;
                let iteration_limit = 100;
                let mut converged = false;
                while !converged && iteration < iteration_limit {

                    // 1. Local update.
                    for i in 0..number_of_nodes {
                        nodes[i].local_x_update();
                        /* println!("Node {} -- x = {} -- z = {} -- u = {} -- c = {}",
                             i, nodes[i].local, nodes[i].global, nodes[i].dual, nodes[i].cost_factor); */
                    }

                    // 2. Gather data.
                    for i in 0..number_of_nodes {
                        solver.variables.x[i] = nodes[i].local;
                        solver.duals.u[i]     = nodes[i].dual;
                    }

                    // 3. Global update.
                    solver.global_z_updater();

                    // 4. Send back data.
                    for i in 0..number_of_nodes {
                        nodes[i].global = solver.globals.x[i];
                    }

                    // 5. Dual update.
                    for i in 0..number_of_nodes {
                        nodes[i].local_dual_update();
                    }

                    // 6. Check if any termination condition is met.
                    let sum_of_globals = f32::sum(solver.globals.x.iter());
                    if (sum_of_globals - 1f32).abs() < tolerance {
                        let mut has_converged = true;
                        for i in 0..number_of_nodes {
                            if (solver.globals.x[i] - solver.variables.x[i]).abs() > (tolerance / number_of_nodes as f32) {
                                has_converged = false;
                            }
                        }
                        converged  = has_converged;
                    }
                    iteration += 1;
                }

                // Save the simulation results.
                samples.push(iteration as f32);
                /* writer.write_record(
                    &[
                        number_of_nodes.to_string(),
                        penalty.to_string(),
                        iteration.to_string(),
                        converged.to_string()
                    ]).expect("Write operation failed. "); */
            }

            // Compute the average of the samples.
            let average_number_of_iterations = samples.iter().sum::<f32>() / number_of_samples as f32;
            iterations_to_converge[penalty]
                .push((number_of_nodes as f32, average_number_of_iterations));
        }
    }

    /* for &number_of_nodes in nodes_range.iter() {

        for &penalty in penalty_range.iter() {

            for &min_cost in min_costs.iter() {

                let variables = Variables { x: vec![0f32; number_of_nodes] };
                let globals   = Variables { x: vec![1f32 / number_of_nodes as f32; number_of_nodes] };
                let duals     = DualsVariables { u: vec![0f32; number_of_nodes] };
                let cost_factors =
                    cost_factors(min_cost, 100f32, number_of_nodes as u32);

                // Initialize the nodes in the system.
                let mut nodes: Vec<ADMMNode> = Vec::new();
                for i in 0..number_of_nodes {
                    nodes.push(ADMMNode {
                        local:       variables.x[i],
                        dual:        duals.u[i],
                        global:      globals.x[i],
                        penalty,
                        cost_factor: cost_factors[i],
                    })
                }

                // And the global solver.
                let mut solver = GlobalSolver {
                    variables,
                    globals,
                    duals,
                };

                // Then simulate a consensus problem resolution.
                let mut iteration = 0;
                let iteration_limit = 100;
                let mut converged = false;
                while !converged && iteration < iteration_limit {

                    // 1. Local update.
                    for i in 0..number_of_nodes {
                        nodes[i].local_x_update();
                        /* println!("Node {} -- x = {} -- z = {} -- u = {} -- c = {}",
                             i, nodes[i].local, nodes[i].global, nodes[i].dual, nodes[i].cost_factor); */
                    }

                    // 2. Gather data.
                    for i in 0..number_of_nodes {
                        solver.variables.x[i] = nodes[i].local;
                        solver.duals.u[i]     = nodes[i].dual;
                    }

                    // 3. Global update.
                    solver.global_z_updater();

                    // 4. Send back data.
                    for i in 0..number_of_nodes {
                        nodes[i].global = solver.globals.x[i];
                    }

                    // 5. Dual update.
                    for i in 0..number_of_nodes {
                        nodes[i].local_dual_update();
                    }

                    // 6. Check if any termination condition is met.
                    let sum_of_globals = f32::sum(solver.globals.x.iter());
                    if (sum_of_globals - 1f32).abs() < tolerance {
                        let mut has_converged = true;
                        for i in 0..number_of_nodes {
                            if (solver.globals.x[i] - solver.variables.x[i]).abs() > (tolerance / number_of_nodes as f32) {
                                has_converged = false;
                            }
                        }
                        converged  = has_converged;
                    }
                    iteration += 1;
                }

                // Save the simulation results.
                writer.write_record(
                    &[
                        number_of_nodes.to_string(),
                        penalty.to_string(),
                        min_cost.to_string(),
                        iteration.to_string(),
                        converged.to_string()
                    ]).expect("Write operation failed. ");

                // Store the results in the vectors for plotting.
                let min_cost_index = min_costs.iter().position(|&i| i == min_cost)
                    .expect("Could not find min_cost in min_costs. ");
                let &penalty_index = &penalty_range.iter().position(|&i| i == penalty)
                    .expect("Could not find penalty in penalty_range. ");
                results_iter[min_cost_index][penalty_index]
                    .push((number_of_nodes as f32, iteration as f32));
                results_conv[min_cost_index][penalty_index]
                .push(converged);
            }

        }

    }*/

    // Plot the results.
    let mut color_list = Vec::new();
    color_list.push(RGBColor(255,0,0));
    color_list.push(RGBColor(240,128,128));
    color_list.push(RGBColor(255,140,0));
    color_list.push(RGBColor(255,255,0));
    color_list.push(RGBColor(0,128,0));
    color_list.push(RGBColor(0,255,127));
    color_list.push(RGBColor(0,255,255));
    color_list.push(RGBColor(0,191,255));
    color_list.push(RGBColor(0,0,255));
    color_list.push(RGBColor(148,0,211));
    color_list.push(RGBColor(255,0,255));
    color_list.push(RGBColor(255,192,203));

    // Print the convergence diagrams.
    let file_dest = "./convergence.png".to_string();

    let root =
        BitMapBackend::new(&file_dest, (2048, 1536)).into_drawing_area();
    root.fill(&WHITE).expect("Failed to fill the figure background. ");

    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(50)
        .y_label_area_size(50)
        //.right_y_label_area_size(40)
        .margin(5)
        .caption("Convergence of ADMM".to_string(), ("sans-serif", 18.0).into_font())
        .build_cartesian_2d(0f32..110f32, 0f32..110f32).expect("Could not build the chart. ");

    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .y_desc("Average number of iteration (to converge)").y_label_style(("sans-serif", 18.0).into_font())
        .x_desc("Number of nodes").x_label_style(("sans-serif", 18.0).into_font())
        .y_label_formatter(&|x| format!("{}", x))
        .x_label_formatter(&|x| format!("{}", x))
        .draw()
        .expect("Could not draw the chart.");

    chart
        .draw_series(LineSeries::new(iterations_to_converge[0].clone(), color_list[0].clone())).expect("Could not draw the chart.")
        .label(format!("Penalty = {}", penalty_range[0]))
        .legend(|(x,y)| {
            Rectangle::new([(x - 15, y + 1), (x, y)], color_list[0])
        });
    chart
        .draw_series(LineSeries::new(iterations_to_converge[1].clone(), color_list[1].clone())).expect("Could not draw the chart.")
        .label(format!("Penalty = {}", penalty_range[1]))
        .legend(|(x,y)| {
            let p = 1;
            Rectangle::new([(x - 15, y + 1), (x, y)], color_list[p])
        });
    chart
        .draw_series(LineSeries::new(iterations_to_converge[2].clone(), color_list[2].clone())).expect("Could not draw the chart.")
        .label(format!("Penalty = {}", penalty_range[2]))
        .legend(|(x,y)| {
            let p = 2;
            Rectangle::new([(x - 15, y + 1), (x, y)], color_list[p])
        });
    chart
        .draw_series(LineSeries::new(iterations_to_converge[3].clone(), color_list[3].clone())).expect("Could not draw the chart.")
        .label(format!("Penalty = {}", penalty_range[3]))
        .legend(|(x,y)| {
            let p = 3;
            Rectangle::new([(x - 15, y + 1), (x, y)], color_list[p])
        });
    chart
        .draw_series(LineSeries::new(iterations_to_converge[4].clone(), color_list[4].clone())).expect("Could not draw the chart.")
        .label(format!("Penalty = {}", penalty_range[4]))
        .legend(|(x,y)| {
            let p = 4;
            Rectangle::new([(x - 15, y + 1), (x, y)], color_list[p])
        });

    chart.configure_series_labels().position(SeriesLabelPosition::UpperRight).margin(20)
        .legend_area_size(5).border_style(BLACK).background_style(WHITE.mix(0.9)).label_font(("Calibri", 20)).draw().unwrap();

    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");

}
