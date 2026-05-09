import rtmidi


def send_cc(canal, cc_number, valor):
    """Envía un Control Change. valor debe estar entre 0 y 127."""
    midi_out.send_message([0xB0 + canal, cc_number, valor])

# Convertir un float (0.0 - 1.0) a rango MIDI (0 - 127)
def to_midi(valor_normalizado):
    return int(max(0, min(127, valor_normalizado * 127)))


midi_out = rtmidi.MidiOut(rtmidi.API_LINUX_ALSA)
ports = midi_out.get_ports()
print(ports)  # Aquí debes ver tu puerto virtual (ej. "loopMIDI Port")
print(ports[0])
midi_out.open_port(0)  # abre directo midisnoop
send_cc(0, 10, 0)
midi_out.send_message([0x90, 60, 100])