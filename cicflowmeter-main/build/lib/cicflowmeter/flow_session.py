import csv
from collections import defaultdict

from scapy.sessions import DefaultSession

from .features.context.packet_direction import PacketDirection
from .features.context.packet_flow_key import get_packet_flow_key
from .flow import Flow

import numpy as np
import pickle

list_columns = ['bwd_pkt_len_std', 'init_bwd_win_byts', 'pkt_len_min', 'fwd_pkt_len_min', 'idle_max', 
    'idle_mean', 'bwd_iat_tot', 'bwd_iat_max', 'bwd_iat_std', 'bwd_pkt_len_max', 'idle_min', 
    'bwd_iat_mean', 'protocol', 'fwd_iat_min', 'flow_iat_min', 'idle_std', 'fwd_pkts/s', 
    'flow_pkts/s', 'bwd_pkts/s', 'flow_byts/s', 'bwd_iat_min', 'active_max', 'active_mean',
    'active_min', 'active_std', 'subflow_bwd_byts', 'totlen_bwd_pkts', 'fin_flag_cnt', 
    'bwd_pkt_len_mean', 'bwd_seg_size_avg', 'tot_bwd_pkts', 'subflow_bwd_pkts', 
    'tot_fwd_pkts', 'subflow_fwd_pkts', 'bwd_header_len', 'fwd_header_len', 'totlen_fwd_pkts', 
    'subflow_fwd_byts', 'fwd_act_data_pkts', 'fwd_iat_tot', 'bwd_pkt_len_min', 'fwd_iat_mean', 
    'fwd_iat_max', 'fwd_iat_std', 'fwd_seg_size_avg', 'fwd_pkt_len_mean', 'flow_iat_mean', 
    'fwd_pkt_len_std', 'fwd_seg_size_min', 'pkt_size_avg', 'pkt_len_mean', 'flow_iat_std', 
    'pkt_len_var', 'flow_duration', 'flow_iat_max', 'pkt_len_std', 'dst_port', 'fwd_pkt_len_max', 
    'init_fwd_win_byts', 'pkt_len_max', 'fwd_psh_flags', 'syn_flag_cnt', 'rst_flag_cnt', 
    'psh_flag_cnt', 'ack_flag_cnt', 'urg_flag_cnt', 'ece_flag_cnt', 'down/up_ratio',
    'fwd_byts/b_avg', 'fwd_pkts/b_avg', 'fwd_blk_rate_avg', 'bwd_byts/b_avg', 
    'bwd_pkts/b_avg', 'bwd_blk_rate_avg']
    
list_label_binary=['Normal','Anomaly']

rf_model = pickle.load(open('RandomForest_Model.pkl', 'rb')) 

EXPIRED_UPDATE = 5
MACHINE_LEARNING_API = "http://localhost:8000/api/catch-log"
GARBAGE_COLLECT_PACKETS = 100

class FlowSession(DefaultSession):
    """Creates a list of network flows."""

    def __init__(self, *args, **kwargs):
        self.flows = {}
        self.csv_line = 0

        if self.output_mode == "flow":
            output = open(self.output_file, "w")
            self.csv_writer = csv.writer(output)

        self.packets_count = 0

        self.clumped_flows_per_label = defaultdict(list)

        super(FlowSession, self).__init__(*args, **kwargs)

    def toPacketList(self):
        # Sniffer finished all the packets it needed to sniff.
        # It is not a good place for this, we need to somehow define a finish signal for AsyncSniffer
        self.garbage_collect(None)
        return super(FlowSession, self).toPacketList()

    def on_packet_received(self, packet):
        count = 0
        direction = PacketDirection.FORWARD

        if self.output_mode != "flow":
            if "TCP" not in packet:
                return
            elif "UDP" not in packet:
                return

        try:
            # Creates a key variable to check
            packet_flow_key = get_packet_flow_key(packet, direction)
            flow = self.flows.get((packet_flow_key, count))
        except Exception:
            return

        self.packets_count += 1

        # If there is no forward flow with a count of 0
        if flow is None:
            # There might be one of it in reverse
            direction = PacketDirection.REVERSE
            packet_flow_key = get_packet_flow_key(packet, direction)
            flow = self.flows.get((packet_flow_key, count))

        if flow is None:
            # If no flow exists create a new flow
            direction = PacketDirection.FORWARD
            flow = Flow(packet, direction)
            packet_flow_key = get_packet_flow_key(packet, direction)
            self.flows[(packet_flow_key, count)] = flow

        elif (packet.time - flow.latest_timestamp) > EXPIRED_UPDATE:
            # If the packet exists in the flow but the packet is sent
            # after too much of a delay than it is a part of a new flow.
            expired = EXPIRED_UPDATE
            while (packet.time - flow.latest_timestamp) > expired:
                count += 1
                expired += EXPIRED_UPDATE
                flow = self.flows.get((packet_flow_key, count))

                if flow is None:
                    flow = Flow(packet, direction)
                    self.flows[(packet_flow_key, count)] = flow
                    break
        elif "F" in str(packet.flags):
            # If it has FIN flag then early collect flow and continue
            flow.add_packet(packet, direction)
            self.garbage_collect(packet.time)
            return

        flow.add_packet(packet, direction)

        if self.packets_count % GARBAGE_COLLECT_PACKETS == 0 or (
            flow.duration > 120 and self.output_mode == "flow"
        ):
            self.garbage_collect(packet.time)

    def get_flows(self) -> list:
        return self.flows.values()

    def garbage_collect(self, latest_time) -> None:
        #TODO: Garbage Collection / Feature Extraction should have a separate thread
        keys = list(self.flows.keys())

        for k in keys:

            flow = self.flows.get(k)

            if (
                latest_time is None
                or latest_time - flow.latest_timestamp > EXPIRED_UPDATE
                or flow.duration > 90
            ):
            
                data = flow.get_data()

                if self.csv_line == 0:
                    self.csv_writer.writerow(data.keys())
        
                X_features = np.array([data[k] for k in list_columns])
                X_features = np.expand_dims(X_features, axis=0)
                pred = rf_model.predict(X_features)[0]
                label = list_label_binary[int(pred)]

                data['label'] = label
                print("type :",label)

                self.csv_writer.writerow(data.values())
                self.csv_line += 1
                
                del self.flows[k]

def generate_session_class(output_mode, output_file, url_model):
    return type(
        "NewFlowSession",
        (FlowSession,),
        {
            "output_mode": output_mode,
            "output_file": output_file,
            "url_model": url_model,
        },
    )
