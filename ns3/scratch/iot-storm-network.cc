#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/tap-bridge-module.h"
#include "ns3/internet-module.h"

#include <cmath>
#include <iostream>
#include <string>

using namespace ns3;

int main(int argc, char *argv[])
{
    uint32_t nIoT = 100;
    double simTime = 3600.0;

    CommandLine cmd;
    cmd.AddValue("nIoT", "Number of IoT nodes", nIoT);
    cmd.AddValue("simTime", "Simulation time in seconds", simTime);
    cmd.Parse(argc, argv);

    uint32_t totalNodes = nIoT + 1;

    GlobalValue::Bind("SimulatorImplementationType",
                      StringValue("ns3::RealtimeSimulatorImpl"));
    GlobalValue::Bind("ChecksumEnabled", BooleanValue(true));

    std::cout << "[NS-3] Nodes: " << totalNodes
              << " (1 gateway + " << nIoT << " IoT)" << std::endl;

    NodeContainer nodes;
    nodes.Create(totalNodes);

    MobilityHelper mobility;
    Ptr<ListPositionAllocator> posAlloc = CreateObject<ListPositionAllocator>();
    posAlloc->Add(Vector(0.0, 0.0, 0.0));
    for (uint32_t i = 1; i < totalNodes; i++) {
        double angle = (i - 1) * (2.0 * M_PI / nIoT);
        double radius = 10.0 + (40.0 * ((i - 1) % 5) / 4.0);
        posAlloc->Add(Vector(radius * std::cos(angle), radius * std::sin(angle), 0.0));
    }
    mobility.SetPositionAllocator(posAlloc);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(nodes);

    YansWifiChannelHelper channel;
    channel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    channel.AddPropagationLoss("ns3::LogDistancePropagationLossModel",
                               "Exponent", DoubleValue(3.0),
                               "ReferenceDistance", DoubleValue(1.0),
                               "ReferenceLoss", DoubleValue(46.67));

    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());
    phy.Set("TxPowerStart", DoubleValue(20.0));
    phy.Set("TxPowerEnd", DoubleValue(20.0));
    phy.Set("RxSensitivity", DoubleValue(-90.0));
    phy.SetErrorRateModel("ns3::NistErrorRateModel");

    WifiMacHelper mac;
    mac.SetType("ns3::AdhocWifiMac");

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211g);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode", StringValue("ErpOfdmRate6Mbps"),
                                 "ControlMode", StringValue("ErpOfdmRate6Mbps"));

    NetDeviceContainer devices = wifi.Install(phy, mac, nodes);

    std::cout << "[NS-3] MAC addresses:" << std::endl;
    for (uint32_t i = 0; i < totalNodes; i++) {
        std::cout << "  tap" << i << " = " << devices.Get(i)->GetAddress() << std::endl;
    }

    InternetStackHelper stack;
    stack.Install(nodes);
    Ipv4AddressHelper addresses;
    addresses.SetBase("10.1.1.0", "255.255.255.0");
    addresses.Assign(devices);

    TapBridgeHelper tapBridge;
    tapBridge.SetAttribute("Mode", StringValue("ConfigureLocal"));
    for (uint32_t i = 0; i < totalNodes; i++) {
        std::string tapName = "tap" + std::to_string(i);
        tapBridge.SetAttribute("DeviceName", StringValue(tapName));
        tapBridge.Install(nodes.Get(i), devices.Get(i));
    }

    Simulator::Stop(Seconds(simTime));

    std::cout << "\n[NS-3] Config: 802.11g 6Mbps, LogDistance n=3 indoor" << std::endl;
    std::cout << "[NS-3] Nodes: 10-50m from gateway, TxPower=20dBm, RxSens=-90dBm" << std::endl;
    std::cout << "[NS-3] Running realtime tap-bridge simulation" << std::endl;

    Simulator::Run();
    Simulator::Destroy();
    return 0;
}
